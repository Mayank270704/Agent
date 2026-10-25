import { afterEach, describe, expect, it, vi } from "vitest";
import { getHealth, sendChatMessage } from "./api";
import { ApiError } from "./types";

function jsonResponse(body: unknown, init?: ResponseInit): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("sendChatMessage", () => {
  it("sends a JSON POST with Content-Type and includes session_id in the body", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ reply: "hi" }));
    vi.stubGlobal("fetch", fetchMock);

    await sendChatMessage({ message: "hello", session_id: "abc-123" });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toContain("/chat");
    expect(init.method).toBe("POST");
    expect(init.headers["Content-Type"]).toBe("application/json");
    const sentBody = JSON.parse(init.body as string);
    expect(sentBody).toEqual({ message: "hello", session_id: "abc-123" });
  });

  it("returns the parsed reply on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ reply: "the answer" })));

    const result = await sendChatMessage({ message: "hi" });

    expect(result).toEqual({ reply: "the answer" });
  });

  it("throws an ApiError using the backend's own string detail on a 4xx/5xx response", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "User message cannot be empty." }, { status: 400 })),
    );

    await expect(sendChatMessage({ message: "" })).rejects.toMatchObject({
      message: "User message cannot be empty.",
      status: 400,
    });
  });

  it("extracts the first message from a Pydantic-style validation error array (422)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        jsonResponse(
          { detail: [{ type: "string_too_long", loc: ["body", "message"], msg: "String should have at most 8000 characters" }] },
          { status: 422 },
        ),
      ),
    );

    await expect(sendChatMessage({ message: "x".repeat(8001) })).rejects.toMatchObject({
      message: "String should have at most 8000 characters",
      status: 422,
    });
  });

  it("never surfaces a raw stack trace or non-JSON body as the error message", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("Internal Server Error", { status: 500, headers: { "Content-Type": "text/plain" } })),
    );

    expect.assertions(3);
    try {
      await sendChatMessage({ message: "hi" });
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).message).not.toContain("Traceback");
      expect((error as ApiError).message).toContain("500");
    }
  });

  it("produces a safe, user-facing error when the network request itself fails", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    await expect(sendChatMessage({ message: "hi" })).rejects.toMatchObject({
      message: "Unable to reach the agent. Check that the backend is running.",
    });
  });

  it("rejects a malformed (non-object / missing reply) success response", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ unexpected: true })));

    await expect(sendChatMessage({ message: "hi" })).rejects.toBeInstanceOf(ApiError);
  });
});

describe("getHealth", () => {
  it("returns the parsed health body on success", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ status: "ok" })));

    expect(await getHealth()).toEqual({ status: "ok" });
  });

  it("still parses a degraded (503) health response rather than treating it as unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({ status: "degraded" }, { status: 503 })));

    expect(await getHealth()).toEqual({ status: "degraded" });
  });

  it("returns null when the backend cannot be reached", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    expect(await getHealth()).toBeNull();
  });
});
