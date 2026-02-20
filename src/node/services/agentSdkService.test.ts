/**
 * Unit tests for AgentSdkService.
 *
 * Tests the SDK adapter service that wraps the Claude Agent SDK's `query()` function
 * and converts SDK messages to mux stream events.
 */

import { describe, test, expect, beforeEach, afterEach, mock } from "bun:test";
import { AgentSdkService } from "./agentSdkService";

describe("AgentSdkService", () => {
  let service: AgentSdkService;

  beforeEach(() => {
    service = new AgentSdkService();
  });

  afterEach(() => {
    mock.restore();
  });

  describe("isAgentSdkModel", () => {
    test("should return true for claude-agent-sdk prefixed models", () => {
      expect(service.isAgentSdkModel("claude-agent-sdk:claude-sonnet-4-6")).toBe(true);
      expect(service.isAgentSdkModel("claude-agent-sdk:claude-opus-4-6")).toBe(true);
      expect(service.isAgentSdkModel("claude-agent-sdk:claude-haiku-3.5")).toBe(true);
    });

    test("should return false for non-SDK models", () => {
      expect(service.isAgentSdkModel("anthropic:claude-sonnet-4-6")).toBe(false);
      expect(service.isAgentSdkModel("openai:gpt-4")).toBe(false);
      expect(service.isAgentSdkModel("claude-sonnet-4-6")).toBe(false);
    });
  });

  describe("isStreaming", () => {
    test("should return false when no active queries", () => {
      expect(service.isStreaming("workspace-1")).toBe(false);
    });
  });

  describe("stopStream", () => {
    test("should return Ok when no active query exists", async () => {
      const result = await service.stopStream("workspace-1");
      expect(result.success).toBe(true);
    });
  });

  describe("event emission", () => {
    test("should allow event subscription", () => {
      // Verify service is an EventEmitter and can subscribe to events
      expect(typeof service.on).toBe("function");
      expect(typeof service.emit).toBe("function");

      const handler = mock(() => undefined);
      service.on("stream-start", handler);

      // Manually emit to verify handler is called
      service.emit("stream-start", { type: "stream-start", workspaceId: "test" });
      expect(handler).toHaveBeenCalledWith({ type: "stream-start", workspaceId: "test" });
    });
  });
});

describe("extractSdkModel utility", () => {
  // Test the model extraction logic indirectly through the service
  const service = new AgentSdkService();

  test("should recognize valid SDK model strings", () => {
    // The isAgentSdkModel method validates the prefix
    expect(service.isAgentSdkModel("claude-agent-sdk:claude-sonnet-4-6")).toBe(true);
    expect(service.isAgentSdkModel("claude-agent-sdk:claude-opus-4-6")).toBe(true);
  });

  test("should reject invalid model strings", () => {
    expect(service.isAgentSdkModel("anthropic:claude-sonnet-4-6")).toBe(false);
    expect(service.isAgentSdkModel("")).toBe(false);
    expect(service.isAgentSdkModel("claude-sonnet-4-6")).toBe(false);
  });
});
