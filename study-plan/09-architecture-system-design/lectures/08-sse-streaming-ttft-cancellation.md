# Lecture 8: Streaming Tokens — SSE vs WebSocket, TTFT, and Upstream Cancellation

> Every chat UI you have ever liked streamed its answer. The words appeared one at a time, and the wait felt short even when the full answer took ten seconds. That perceived speed is a *measurable* engineering quantity — **time to first token (TTFT)** — and it is produced by a transport you can get right or catastrophically wrong. This lecture makes streaming *correct* and *cheap*. You will learn why **SSE is the right default** for token streaming and when WebSocket is worth its cost; how to measure TTFT *separately* from total time and why that split is load-bearing; how to build a `text/event-stream` endpoint in FastAPI that forwards provider chunks; and the one bug that silently burns money in production — **failing to abort the upstream provider call when the client disconnects**. After this lecture you can build a streaming endpoint that frames events correctly, survives proxies, and stops the token meter the instant nobody is reading.

**Prerequisites:** Solid HTTP/async fundamentals (`async`/`await`, ASGI, chunked transfer, status codes); comfort with FastAPI and `httpx`; you've called a streaming LLM API at least once. · **Reading time:** ~30 min · **Part of:** AI Application Architecture & System Design — Week 2

## The core idea (plain language)

An LLM generates output one token at a time, left to right. The model produces token 1, then token 2, and so on — each one takes a few milliseconds of GPU time. If your server waits for the *entire* completion before responding, the user stares at a spinner for the full generation time (often 3–15 seconds for a long answer). If instead you forward each token to the browser *as it is produced*, the user sees words appear almost immediately and reads along while generation continues. Same total time; completely different felt experience.

Two facts drive every decision in this lecture:

1. **Users feel the first token, not the last one.** The gap between "user hit send" and "first word appears" is what makes an app feel fast or sluggish. This is TTFT. It is a different number from total latency and you must measure it on its own.
2. **The provider bills you for every token it generates, whether or not anyone reads it.** If the user closes the tab mid-answer and you don't tell the provider to stop, the model keeps generating — and you keep paying — until it hits the natural end or `max_tokens`. Streaming *without cancellation* is a money leak.

The transport question — SSE vs WebSocket — is really a question of *directionality*. Token streaming is **one-way**: server pushes tokens to the client, and the client has nothing to say back mid-stream. That is exactly what **Server-Sent Events (SSE)** is for: a long-lived HTTP response that emits a sequence of text events. WebSocket gives you a **bidirectional** channel, which you need for voice or live collaboration — but it comes with real costs (harder proxying, manual reconnection, connection state) that you should not pay to solve a one-way problem.

## How it actually works (mechanism, from first principles)

### SSE is just a long HTTP response that never stops (until it does)

There is no magic in SSE. It is a normal HTTP `GET` whose response has `Content-Type: text/event-stream` and a body that the server writes to incrementally and never closes until the stream ends. The connection uses HTTP chunked transfer encoding under the hood, so bytes flush to the client as you produce them.

The wire format is a tiny line-based protocol. An "event" is one or more `field: value` lines, terminated by a **blank line**:

```
data: Hello

data: world

event: done
data: [DONE]

```

Rules that bite you if you get them wrong:

- Each event ends with a **double newline** (`\n\n`). One missing blank line and the browser buffers your event forever, waiting for the terminator. TTFT looks infinite.
- `data:` is the payload. Multiple `data:` lines in one event are concatenated with newlines by the client.
- `event:` names a custom event type (the client listens with `addEventListener("done", ...)`); omit it and the client fires the default `message` event.
- `id:` sets the last-event-id, which the browser echoes back in the `Last-Event-ID` header on reconnect — this is what powers **automatic reconnection and resume** for free.
- Lines starting with `:` are comments. `: keep-alive\n\n` is the standard **heartbeat** — it keeps idle intermediaries from closing the connection without emitting a visible event.

On the client, the entire protocol is one built-in browser object:

```js
const es = new EventSource("/chat/stream?conversation_id=abc");
es.onmessage = (e) => appendToken(e.data);      // default "message" events
es.addEventListener("done", () => es.close());   // your custom terminator
es.onerror = () => { /* browser auto-reconnects with Last-Event-ID */ };
```

That auto-reconnect is the killer feature. If the network blips, the browser reconnects on its own and sends `Last-Event-ID` so you can resume. With WebSocket you write all of that yourself.

### WebSocket and the cost of choosing wrong

WebSocket starts as an HTTP request with an `Upgrade: websocket` header, then the TCP connection is "upgraded" to a full-duplex binary channel that is no longer HTTP. Both sides can send frames at any time. That is genuinely necessary for:

- **Voice / realtime audio** — you stream microphone audio *up* while receiving audio *down*, simultaneously.
- **Live collaboration** — multiple clients edit shared state and every keystroke flows both ways.
- **Interactive control mid-generation** — e.g., a client that steers or interrupts a running agent with structured messages (though even "interrupt" is often better done as a separate small HTTP call).

The cost of reaching for WebSocket when you only needed SSE:

| Concern | SSE | WebSocket |
|---|---|---|
| Reconnect on drop | automatic, with resume | you build it |
| Works through HTTP proxies/CDNs | yes (it's just HTTP) | often needs special config; some corporate proxies block upgrades |
| Load balancer / HTTP/2 friendliness | native | sticky-session and upgrade handling required |
| Auth (cookies, bearer headers) | normal HTTP auth | headers on the handshake only; awkward from browsers |
| Server complexity | a generator function | connection lifecycle, ping/pong, backpressure state |

Choosing WebSocket for plain token streaming means you inherit reconnection logic, connection-state bugs, and proxy headaches to solve a problem SSE solves for free. **Default to SSE; upgrade to WebSocket only when the data genuinely flows both ways at once.**

### TTFT vs total time — and why you measure them separately

Draw the timeline of one streaming request:

```
t0        t1                                      t2
|--------->|--------------------------------------->|
 request    FIRST token arrives            LAST token arrives
 sent       (TTFT = t1 - t0)               (total = t2 - t0)
```

- **TTFT = t1 − t0.** Dominated by: network RTT to the provider, the provider's queue wait, and the model's *prefill* — the time to process your entire prompt before it can emit token 1. A long prompt (big RAG context) inflates TTFT even if the answer is short.
- **Total = t2 − t0.** TTFT plus the *decode* phase: (output tokens − 1) × inter-token time.

These two numbers move for *different reasons*, so a single "latency" metric hides the diagnosis. Consider two requests that both take 4.0 s total:

- Request A: TTFT 3.5 s, then 200 tokens in 0.5 s. The user watched a spinner for 3.5 s. **Feels terrible.** The problem is prefill/queueing — shrink the prompt, cache the prefix, or fix routing.
- Request B: TTFT 0.3 s, then 200 tokens in 3.7 s. The user started reading in 300 ms. **Feels great.** The "slowness" is just a long answer streaming in, which reading time hides.

Same total, opposite user experience, opposite fix. If you only logged total time you could not tell them apart. **Always record `ttft_ms` and `total_ms` per request, separately.** The measurement is trivial: mark `t0` when you start, capture `t1` at the first yielded chunk, `t2` at the end.

```python
import time

async def timed_stream(chunks):
    t0 = time.perf_counter()
    ttft_ms = None
    n_out = 0
    async for chunk in chunks:
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t0) * 1000
        n_out += 1
        yield chunk
    total_ms = (time.perf_counter() - t0) * 1000
    # persist ttft_ms, total_ms, n_out to your metrics store here
```

Track p50 **and p95** for each. TTFT p95 is where prefill spikes and provider queueing hide; averages lie because a few 6-second cold starts vanish into the mean.

### A minimal FastAPI SSE endpoint that forwards provider chunks

`StreamingResponse` with `media_type="text/event-stream"` is the whole mechanism. You pass it an async generator; FastAPI iterates it and flushes each yield.

```python
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

def sse(data: str, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {data}\n\n"      # note the DOUBLE newline

async def token_source(prompt: str):
    # provider_stream yields incremental text chunks (e.g. OpenAI/Anthropic deltas)
    async for chunk in provider_stream(prompt):
        yield chunk

@app.get("/chat/stream")
async def chat_stream(request: Request, q: str):
    async def gen():
        async for tok in token_source(q):
            yield sse(json.dumps({"text": tok}))
        yield sse("[DONE]", event="done")
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # tell nginx: do NOT buffer this response
            "Connection": "keep-alive",
        },
    )
```

Two headers earn their place: `Cache-Control: no-cache` stops intermediaries caching a partial stream, and `X-Accel-Buffering: no` is the nginx directive that disables response buffering for *this route only* (more on that under proxies).

### The load-bearing bug: aborting the upstream call on client disconnect

Here is the failure that this entire lecture builds toward. Your generator forwards tokens from a provider. The user closes the tab after 2 seconds. What happens to the provider call?

If you do nothing special: **the provider keeps generating tokens.** The `httpx` stream to the provider is still open, the model still decodes, and your bill still climbs — for output no human will ever see. On a long `max_tokens=4000` completion at, say, $15 / 1M output tokens, each abandoned request that runs to completion instead of stopping at 2 s of reading can waste the majority of ~4000 output tokens. Multiply by every impatient user who hits "stop" or navigates away, and it is a real line item.

The fix has two halves that you must wire together:

**1. Detect the disconnect.** FastAPI exposes `await request.is_disconnected()`, which returns `True` once the client's TCP connection is gone. Poll it in the loop:

```python
async for tok in token_source(q):
    if await request.is_disconnected():
        break            # stop reading from upstream
    yield sse(json.dumps({"text": tok}))
```

But `break` alone is not enough. You stopped *reading*, but did you stop the *provider from producing*? That depends on how the upstream generator is structured. You must ensure the break propagates all the way to closing the upstream HTTP stream.

**2. Cancel the upstream so it actually closes.** The clean pattern is a structured `try/finally` that closes the provider stream (or cancels its task) no matter how the generator exits — normal completion, client disconnect, or exception:

```python
import httpx

async def token_source(prompt: str):
    client = httpx.AsyncClient()
    try:
        async with client.stream("POST", PROVIDER_URL, json=payload) as resp:
            async for line in resp.aiter_lines():
                tok = parse_provider_line(line)
                if tok is not None:
                    yield tok
    finally:
        await client.aclose()   # closes the socket -> upstream stops generating
```

When the consumer's `break` (or a raised `CancelledError`) unwinds this generator, Python runs the `finally`, `client.aclose()` tears down the TCP connection to the provider, and the provider sees the reader is gone and stops decoding. **That socket close is what stops the token meter.**

An even more robust variant runs the upstream pump in a separate `asyncio.Task` and cancels it explicitly, which guarantees teardown even if the consumer never resumes the generator:

```python
import asyncio

async def chat_stream(request: Request, q: str):
    queue: asyncio.Queue = asyncio.Queue()

    async def pump():
        try:
            async for tok in token_source(q):
                await queue.put(tok)
        finally:
            await queue.put(None)   # sentinel = end

    task = asyncio.create_task(pump())

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                tok = await queue.get()
                if tok is None:
                    break
                yield sse(json.dumps({"text": tok}))
        finally:
            task.cancel()                 # kills the upstream pump
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(gen(), media_type="text/event-stream")
```

`task.cancel()` raises `CancelledError` inside `pump`, which unwinds `token_source`, which runs its `finally`, which closes the httpx stream. The chain is: **client gone → cancel task → CancelledError → generator finally → socket close → provider stops.**

**How you prove it stopped.** This is a correctness property, so assert it, don't assume it. Instrument the upstream loop with a counter of tokens *received from the provider* (not tokens sent to the client). In a test: start a stream against a fake provider that emits a token every 50 ms up to 100 tokens; disconnect the client after ~5 tokens; then assert the provider-side counter **stops advancing** shortly after disconnect (e.g., it never reaches 100, and its final value is small). If cancellation is broken, the counter marches to 100 even though the client left — that's the bug, caught.

## Worked example

A support-chat endpoint. Prompt is a 3,000-token RAG context; the answer is up to 500 output tokens. Assume: provider prefill ≈ 300 ms for 3k tokens, inter-token time ≈ 18 ms, network RTT ≈ 40 ms, and output price $15 / 1M tokens.

**Non-streaming (wait for full answer):**
- User waits: 40 ms RTT + 300 ms prefill + 500 × 18 ms decode + 40 ms = **9.38 s** of spinner before *anything* appears.

**Streaming, healthy path:**
- TTFT ≈ 40 + 300 + 18 = **~358 ms** — first word in about a third of a second.
- Total ≈ 358 ms + 499 × 18 ms ≈ **9.34 s**, but the user was reading the whole time.
- Same total; TTFT dropped from 9.38 s to 0.36 s. That is the entire point of streaming.

**Now the cancellation math.** Suppose 20% of users close the tab after reading ~50 tokens (they got their answer early or bailed). For those requests:
- *With cancellation:* provider generates ~50 tokens, then stops. Output cost ≈ 50 tokens.
- *Without cancellation:* provider runs to the full 500 tokens. Output cost ≈ 500 tokens — **10× the output tokens for that request, all wasted.**

Across 100k requests/day with 20% early-exit, that's 20k requests each wasting ~450 output tokens = 9M wasted tokens/day ≈ **$135/day ≈ $4,050/month** burned on tokens nobody read — from a single missing `finally: await client.aclose()`. (Numbers illustrative; the durable lesson is the *shape*: a 5–10× waste on every early-exit request.)

## How it shows up in production

- **"Streaming works locally but TTFT is 8 s in prod."** Almost always **proxy buffering**. nginx (`proxy_buffering on` by default) and some CDNs accumulate the response and forward it in big chunks — or only when the response completes — which converts your beautiful token-by-token stream into one blob at the end. The fix: disable buffering *for the stream route only* — `proxy_buffering off;` (or send `X-Accel-Buffering: no`), plus `proxy_read_timeout` high enough for long generations, and ensure HTTP/1.1 with `Connection ""` on the upstream. On Cloudflare, buffering/optimization features (and some WAF/compression paths) can also stall SSE; test the real edge, not just localhost.
- **The silent money leak.** Missing upstream cancellation doesn't throw, doesn't log, doesn't page anyone. It just shows up as a token bill higher than your request volume predicts. You find it by comparing *tokens billed by the provider* against *tokens delivered to clients* — a persistent gap means abandoned generations are running to completion.
- **Idle-connection deaths.** If the model "thinks" for a while (long prefill, or a tool call before the first token), an idle SSE connection can be closed by a load balancer or proxy after ~30–60 s. **Heartbeats** (`: ping\n\n` every 15–20 s) keep it alive and also let you detect disconnect sooner (a failed write reveals the client is gone).
- **Buffered event framing bugs.** Forget the double newline, or emit a lone `\n`, and the client's parser waits for a terminator that never comes — TTFT appears infinite even though bytes are flowing. Test the raw bytes, not just "does the browser show text."
- **Errors mid-stream.** Once you've sent a `200 OK` and started streaming, you *cannot* change the status code. A provider failure at token 300 can't become an HTTP 500. You must emit an in-band error event (`event: error\ndata: {...}\n\n`) and let the client render it. Design your event schema with an explicit `error` event from day one.
- **Reconnect storms.** SSE auto-reconnects — great, until a provider outage causes thousands of clients to reconnect in a tight loop. Honor the `retry:` field to set a backoff and don't let reconnects re-trigger expensive generations without idempotency.

## Common misconceptions & failure modes

- **"SSE is old/limited; WebSocket is the modern choice."** No. For one-way server→client push, SSE is *simpler and more robust* (auto-reconnect, plain HTTP, proxy-friendly). WebSocket is not an upgrade; it's a different tool for bidirectional traffic. Using it for token streaming adds cost with no benefit.
- **"`break`-ing out of my loop stops the provider."** Only if the break unwinds to code that closes the upstream socket. Without a `try/finally` (or task cancel) that calls `aclose()`, the provider connection can linger and keep generating. Prove it with a token counter test.
- **"`request.is_disconnected()` fires the instant the user leaves."** It's detected on the next check, and for a stalled/idle connection it may only surface when you next try to *write* and the write fails. Combine polling with heartbeats so you notice reasonably fast.
- **"TTFT and total latency are basically the same."** They diverge hard: a huge prompt gives high TTFT with fast decode; a short prompt with a long answer gives tiny TTFT and high total. They have different causes and different fixes — measure both.
- **"Streaming makes requests cheaper."** Streaming changes *when* tokens arrive, not *how many* you're billed for. It's *cancellation* that saves money, by cutting generation short when nobody's reading.
- **"I set `X-Accel-Buffering: no`, so buffering is solved."** That's the nginx hint; a CDN, a second proxy hop, response compression, or a framework middleware can still buffer. Verify end-to-end from the real client through the real edge.

## Rules of thumb / cheat sheet

- **Default transport for token streaming: SSE.** Reach for WebSocket only when data flows *both ways at once* (voice, live collab).
- **Always log `ttft_ms` and `total_ms` separately;** track p50 and p95 of each. High TTFT p95 → prefill/queue/routing problem. High total with low TTFT → just a long answer, usually fine.
- **Every SSE event ends with `\n\n`.** Send a heartbeat comment (`: ping\n\n`) every ~15–20 s to survive idle timeouts.
- **Wrap upstream calls in `try/finally: await client.aclose()`** (or run them in a task you `cancel()`), so client disconnect closes the provider socket.
- **Poll `await request.is_disconnected()` in the stream loop and `break`;** ensure the break reaches the `finally` that closes upstream.
- **Prove cancellation with a test:** count tokens received *from the provider*, disconnect early, assert the count stops rising.
- **Disable proxy buffering for the stream route only** (`proxy_buffering off;` / `X-Accel-Buffering: no`) and raise read timeouts; test through the real proxy/CDN.
- **Design an in-band `error` event now** — you can't send an HTTP error after the stream starts.
- **Set `retry:` for sane reconnect backoff;** make regenerations idempotent so reconnect storms don't multiply cost.

## Connect to the lab

This lecture powers **Week 2, Lab step 3** (`app/stream.py`): a `GET /chat/stream` endpoint returning `text/event-stream`, forwarding provider tokens as SSE events, recording TTFT and total time per request into Postgres. The correctness gate in the Definition of Done — "a client disconnect verifiably aborts the upstream call (token production stops — shown in logs/test)" — is exactly the token-counter test described above; wire it into `tests/test_stream_cancel.py`. When you deploy behind nginx or Cloudflare later in Week 3, revisit the proxy-buffering section before you trust your production TTFT numbers.

## Going deeper (optional)

- **MDN Web Docs — "Using server-sent events"** and the **`EventSource`** reference (`developer.mozilla.org`). The canonical, correct description of the wire format, reconnection, and `Last-Event-ID`.
- **WHATWG HTML Living Standard — Server-Sent Events section** (`html.spec.whatwg.org`). The actual spec if you need to settle a framing dispute.
- **Starlette docs — `StreamingResponse`** and **FastAPI docs** (`fastapi.tiangolo.com`) for the ASGI streaming and `request.is_disconnected()` semantics. Search: "FastAPI StreamingResponse disconnect".
- **`httpx` docs — streaming responses / `client.stream`** (`www.python-httpx.org`). How to consume and, crucially, *close* an upstream stream.
- **nginx docs — `proxy_buffering` and `X-Accel-Buffering`** (`nginx.org`). The definitive word on disabling buffering per-location.
- Provider streaming references: **OpenAI API streaming** and **Anthropic Messages streaming** docs for the exact chunk/delta and `[DONE]`/`message_stop` semantics you'll parse. Search: "OpenAI streaming chat completions", "Anthropic messages streaming events".
- Search queries: "SSE vs WebSocket token streaming", "abort upstream LLM call on client disconnect FastAPI", "TTFT time to first token measurement".

## Check yourself

1. Token streaming is one-way server→client. Give two concrete features that *do* justify WebSocket, and name two costs you take on by using WebSocket where SSE would have sufficed.
2. Two requests each take 4.0 s total. One has TTFT 3.6 s, the other 0.3 s. Which feels faster, why, and what's the likely fix for the slow one?
3. Your generator does `if await request.is_disconnected(): break` when the client leaves, yet your provider bill keeps climbing on abandoned requests. What's missing, and where exactly does the fix go?
4. Streaming works perfectly on localhost but TTFT is ~7 s in production. Name the most likely cause and the specific fix, scoped correctly.
5. How do you *prove*, in an automated test, that a client disconnect actually stopped upstream token production — not just that your server stopped sending?
6. A provider error occurs at token 300 of a stream that already returned HTTP 200. Why can't you return a 500, and what do you do instead?

### Answer key

1. Justified by **bidirectional, simultaneous** data: e.g., live voice (mic audio up while audio streams down) and live multi-user collaboration (every keystroke both ways). Costs of misusing WebSocket for one-way streaming: you must hand-build reconnection/resume (SSE gives it free via `Last-Event-ID`), and you inherit proxy/CDN/load-balancer friction (upgrade handling, sticky sessions) that plain HTTP SSE avoids. Also awkward auth and manual connection-state management.
2. The **0.3 s TTFT** request feels far faster — the user starts reading in 300 ms; the other stares at a spinner for 3.6 s. The slow one's problem is prefill/queueing (long prompt or provider queue), so the fix is shrinking the prompt/context, prompt-prefix caching, or better routing — not touching decode speed.
3. `break` stops you *reading*, but the **upstream httpx stream is never closed**, so the provider keeps generating. The fix is a `try/finally: await client.aclose()` around the upstream `client.stream(...)` (or running the pump in a task you `task.cancel()`), so the unwinding generator closes the provider socket. It goes in the upstream `token_source`, wrapping the provider call.
4. **Proxy/CDN buffering** (nginx `proxy_buffering on` by default, or a CDN accumulating the response). Fix, scoped to the stream route only: `proxy_buffering off;` / send `X-Accel-Buffering: no`, raise `proxy_read_timeout`, ensure HTTP/1.1 keep-alive. Verify through the real edge, not localhost.
5. Instrument a counter of tokens **received from the provider** (upstream side, not client-delivered). In the test, use a fake provider that emits tokens on a timer up to N; disconnect the client early; then assert the upstream counter **stops advancing** and never reaches N. If cancellation is broken, it climbs to N despite the client being gone.
6. Once `200 OK` and the response body have started, the status code is already committed on the wire and can't be changed. You emit an **in-band error event** (`event: error\ndata: {...}\n\n`) that the client is coded to recognize and render, then close the stream — which is why you design an explicit `error` event into the SSE schema from the start.
