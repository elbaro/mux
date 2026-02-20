/**
 * Agent SDK Service - Adapter for Claude Agent SDK execution
 *
 * The Claude Agent SDK is a fundamentally different execution model from other providers:
 * - Other providers: `streamText(model)` → mux handles tools/agent loop
 * - Agent SDK: `query(prompt)` → SDK handles tools/agent loop internally
 *
 * This service wraps the SDK's `query()` function and converts SDK messages to mux
 * stream events so they can be displayed in the mux UI.
 *
 * NOTE: Types from `@anthropic-ai/claude-agent-sdk` are defined locally rather than
 * imported because the SDK's type declarations reference `@anthropic-ai/sdk` types
 * (BetaMessage, BetaRawMessageStreamEvent, etc.) which are not installed. Importing
 * them would cause all dependent types to resolve to `error`.
 */

import { EventEmitter } from "events";

import type { Result } from "@/common/types/result";
import { Ok, Err } from "@/common/types/result";
import type { SendMessageError } from "@/common/types/errors";
import type {
  StreamStartEvent,
  StreamDeltaEvent,
  StreamEndEvent,
  StreamAbortEvent,
  ToolCallStartEvent,
  ToolCallEndEvent,
  ReasoningDeltaEvent,
  UsageDeltaEvent,
  CompletedMessagePart,
} from "@/common/types/stream";
import { log } from "./log";
import { createAssistantMessageId } from "./utils/messageIds";

// ---------------------------------------------------------------------------
// Local SDK Types
//
// Mirrors of the SDK's type shapes needed for message processing.
// Defined locally to avoid unresolvable type dependencies on @anthropic-ai/sdk.
// ---------------------------------------------------------------------------

type SdkPermissionMode = "default" | "acceptEdits" | "bypassPermissions" | "plan" | "dontAsk";

interface SdkUsage {
  input_tokens: number;
  output_tokens: number;
}

interface SdkAssistantMessagePayload {
  content: ContentBlock[];
  usage: SdkUsage;
}

interface SdkSystemMessage {
  type: "system";
  subtype: string;
  tools?: string[];
  model?: string;
}

/** Stream events carry partial content as Anthropic raw message stream events. */
interface SdkStreamEvent {
  type: string;
  delta?: { type: string; text?: string; thinking?: string };
  content_block?: ContentBlock;
}

interface SdkPartialAssistantMessage {
  type: "stream_event";
  event: SdkStreamEvent;
}

interface SdkAssistantMessage {
  type: "assistant";
  message: SdkAssistantMessagePayload;
}

interface SdkResultMessage {
  type: "result";
  subtype: string;
  num_turns: number;
  total_cost_usd: number;
  usage: SdkUsage;
}

interface SdkToolProgressMessage {
  type: "tool_progress";
  tool_name: string;
  elapsed_time_seconds: number;
}

type SdkMessage =
  | SdkSystemMessage
  | SdkPartialAssistantMessage
  | SdkAssistantMessage
  | SdkResultMessage
  | SdkToolProgressMessage
  | { type: string };

interface SdkOptions {
  cwd?: string;
  model?: string;
  includePartialMessages?: boolean;
  permissionMode?: SdkPermissionMode;
  allowedTools?: string[];
  disallowedTools?: string[];
  systemPrompt?: string | { type: "preset"; preset: "claude_code"; append?: string };
  env?: Record<string, string | undefined>;
  abortController?: AbortController;
}

interface SdkQuery extends AsyncGenerator<SdkMessage, void> {
  interrupt(): Promise<void>;
  close(): void;
}

type SdkQueryFn = (args: { prompt: string; options: SdkOptions }) => SdkQuery;

// ---------------------------------------------------------------------------
// SDK Provider Config
// ---------------------------------------------------------------------------

/**
 * SDK-specific provider configuration options.
 * These are stored in providers.jsonc under the "claude-agent-sdk" key.
 */
export interface AgentSdkProviderConfig {
  /** API key for Anthropic (can also come from ANTHROPIC_API_KEY env var) */
  apiKey?: string;
  /** Model to use (e.g., "claude-sonnet-4-6", "claude-opus-4-6") */
  model?: string;
  /** Permission mode for tool execution */
  permissionMode?: SdkPermissionMode;
  /** Additional allowed tools beyond defaults */
  allowedTools?: string[];
  /** Tools to disable */
  disallowedTools?: string[];
  /** System prompt configuration */
  systemPrompt?: string | { type: "preset"; preset: "claude_code"; append?: string };
}

// ---------------------------------------------------------------------------
// Stream Options
// ---------------------------------------------------------------------------

export interface AgentSdkStreamOptions {
  /** User prompt to send */
  prompt: string;
  /** Workspace ID for event routing */
  workspaceId: string;
  /** Working directory for the SDK */
  cwd: string;
  /** Model string (e.g., "claude-agent-sdk:claude-sonnet-4-6") */
  modelString: string;
  /** Provider config from providers.jsonc */
  providerConfig: AgentSdkProviderConfig;
  /** Abort signal for cancellation */
  abortSignal?: AbortSignal;
  /** API key (resolved from config or env) */
  apiKey?: string;
}

// ---------------------------------------------------------------------------
// SDK Content Block Types (from Anthropic API)
// ---------------------------------------------------------------------------

interface TextBlock {
  type: "text";
  text: string;
}

interface ToolUseBlock {
  type: "tool_use";
  id: string;
  name: string;
  input: unknown;
}

interface ThinkingBlock {
  type: "thinking";
  thinking: string;
  signature?: string;
}

type ContentBlock = TextBlock | ToolUseBlock | ThinkingBlock | { type: string };

// ---------------------------------------------------------------------------
// Message Conversion
// ---------------------------------------------------------------------------

/**
 * Extract the model name from a model string like "claude-agent-sdk:claude-sonnet-4-6"
 */
function extractSdkModel(modelString: string): string {
  const parts = modelString.split(":");
  // If it has a provider prefix, return everything after the first colon
  if (parts.length > 1 && parts[0] === "claude-agent-sdk") {
    return parts.slice(1).join(":");
  }
  // Otherwise return as-is
  return modelString;
}

/**
 * Convert SDK content blocks to mux message parts.
 */
function convertContentBlocksToParts(contentBlocks: ContentBlock[]): CompletedMessagePart[] {
  const parts: CompletedMessagePart[] = [];
  const now = Date.now();

  for (const block of contentBlocks) {
    if (block.type === "text") {
      const textBlock = block as TextBlock;
      parts.push({
        type: "text",
        text: textBlock.text,
        timestamp: now,
      });
    } else if (block.type === "tool_use") {
      const toolBlock = block as ToolUseBlock;
      parts.push({
        type: "dynamic-tool",
        toolCallId: toolBlock.id,
        toolName: toolBlock.name,
        input: toolBlock.input,
        state: "input-available", // SDK handles tool execution internally
        timestamp: now,
      });
    } else if (block.type === "thinking") {
      const thinkingBlock = block as ThinkingBlock;
      parts.push({
        type: "reasoning",
        text: thinkingBlock.thinking,
        timestamp: now,
        signature: thinkingBlock.signature,
      });
    }
  }

  return parts;
}

// ---------------------------------------------------------------------------
// Type guards for SDK messages
// ---------------------------------------------------------------------------

function isSdkSystemMessage(msg: SdkMessage): msg is SdkSystemMessage {
  return msg.type === "system";
}

function isSdkPartialMessage(msg: SdkMessage): msg is SdkPartialAssistantMessage {
  return msg.type === "stream_event";
}

function isSdkAssistantMessage(msg: SdkMessage): msg is SdkAssistantMessage {
  return msg.type === "assistant";
}

function isSdkToolProgressMessage(msg: SdkMessage): msg is SdkToolProgressMessage {
  return msg.type === "tool_progress";
}

function isSdkResultMessage(msg: SdkMessage): msg is SdkResultMessage {
  return msg.type === "result";
}

// ---------------------------------------------------------------------------
// Agent SDK Service
// ---------------------------------------------------------------------------

export class AgentSdkService extends EventEmitter {
  private activeQueries = new Map<string, SdkQuery>();

  constructor() {
    super();
    this.setMaxListeners(50);
  }

  /**
   * Check if a model string represents the Claude Agent SDK provider.
   */
  isAgentSdkModel(modelString: string): boolean {
    return modelString.startsWith("claude-agent-sdk:");
  }

  /**
   * Stream a message using the Claude Agent SDK.
   *
   * This is the main entry point for SDK execution. It:
   * 1. Calls the SDK's `query()` function
   * 2. Iterates over SDK messages
   * 3. Converts them to mux stream events
   * 4. Emits events for the UI to display
   */
  async streamWithSdk(options: AgentSdkStreamOptions): Promise<Result<void, SendMessageError>> {
    const { prompt, workspaceId, cwd, modelString, providerConfig, abortSignal, apiKey } = options;

    const sdkModel = extractSdkModel(modelString);
    const messageId = createAssistantMessageId();
    const startTime = Date.now();

    log.info("[AgentSdkService] Starting SDK stream", {
      workspaceId,
      model: sdkModel,
      cwd,
    });

    try {
      // Dynamic import: SDK is heavyweight (spawns subprocesses) and only needed
      // when the user selects "Claude Agent SDK" as their provider.
      // eslint-disable-next-line no-restricted-syntax
      const sdk = (await import("@anthropic-ai/claude-agent-sdk")) as { query: SdkQueryFn };
      const { query } = sdk;

      // Build SDK options
      const sdkOptions: SdkOptions = {
        cwd,
        model: sdkModel,
        includePartialMessages: true, // Enable streaming
        permissionMode: providerConfig.permissionMode ?? "acceptEdits",
        allowedTools: providerConfig.allowedTools,
        disallowedTools: providerConfig.disallowedTools,
        systemPrompt: providerConfig.systemPrompt,
        // Pass API key via env if provided
        env: apiKey ? { ...process.env, ANTHROPIC_API_KEY: apiKey } : process.env,
      };

      // Create abort controller for SDK
      const abortController = new AbortController();
      if (abortSignal) {
        abortSignal.addEventListener("abort", () => {
          abortController.abort();
        });
      }
      sdkOptions.abortController = abortController;

      // Start the query
      const queryIterator = query({ prompt, options: sdkOptions });
      this.activeQueries.set(workspaceId, queryIterator);

      // Emit stream-start event
      const streamStartEvent: StreamStartEvent = {
        type: "stream-start",
        workspaceId,
        messageId,
        model: modelString,
        historySequence: 0, // Will be assigned by history service
        startTime,
      };
      this.emit("stream-start", streamStartEvent);

      // Track accumulated content for stream-end
      let totalInputTokens = 0;
      let totalOutputTokens = 0;
      const parts: CompletedMessagePart[] = [];

      // Process SDK messages
      for await (const sdkMessage of queryIterator) {
        if (abortSignal?.aborted) {
          break;
        }

        if (isSdkSystemMessage(sdkMessage)) {
          if (sdkMessage.subtype === "init") {
            log.debug("[AgentSdkService] SDK initialized", {
              workspaceId,
              tools: sdkMessage.tools,
              model: sdkMessage.model,
            });
          }
        } else if (isSdkPartialMessage(sdkMessage)) {
          this.handleStreamEvent(sdkMessage, workspaceId, messageId);
        } else if (isSdkAssistantMessage(sdkMessage)) {
          this.handleAssistantMessage(sdkMessage, workspaceId, messageId, parts);
          totalInputTokens += sdkMessage.message.usage.input_tokens;
          totalOutputTokens += sdkMessage.message.usage.output_tokens;
        } else if (isSdkToolProgressMessage(sdkMessage)) {
          log.debug("[AgentSdkService] Tool progress", {
            workspaceId,
            toolName: sdkMessage.tool_name,
            elapsed: sdkMessage.elapsed_time_seconds,
          });
        } else if (isSdkResultMessage(sdkMessage)) {
          log.info("[AgentSdkService] SDK query completed", {
            workspaceId,
            subtype: sdkMessage.subtype,
            numTurns: sdkMessage.num_turns,
            totalCostUsd: sdkMessage.total_cost_usd,
          });

          // Emit usage-delta with final usage
          const usageEvent: UsageDeltaEvent = {
            type: "usage-delta",
            workspaceId,
            messageId,
            usage: {
              inputTokens: sdkMessage.usage.input_tokens,
              outputTokens: sdkMessage.usage.output_tokens,
              totalTokens: sdkMessage.usage.input_tokens + sdkMessage.usage.output_tokens,
            },
            cumulativeUsage: {
              inputTokens: sdkMessage.usage.input_tokens,
              outputTokens: sdkMessage.usage.output_tokens,
              totalTokens: sdkMessage.usage.input_tokens + sdkMessage.usage.output_tokens,
            },
          };
          this.emit("usage-delta", usageEvent);
        } else {
          log.debug("[AgentSdkService] Unhandled SDK message type", {
            workspaceId,
            type: sdkMessage.type,
          });
        }
      }

      // Emit stream-end event
      const endEvent: StreamEndEvent = {
        type: "stream-end",
        workspaceId,
        messageId,
        metadata: {
          model: modelString,
          usage: {
            inputTokens: totalInputTokens,
            outputTokens: totalOutputTokens,
            totalTokens: totalInputTokens + totalOutputTokens,
          },
          duration: Date.now() - startTime,
        },
        parts,
      };
      this.emit("stream-end", endEvent);

      this.activeQueries.delete(workspaceId);
      return Ok(undefined);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log.error("[AgentSdkService] SDK stream error", { workspaceId, error: errorMessage });

      // Emit stream-abort on error
      const abortEvent: StreamAbortEvent = {
        type: "stream-abort",
        workspaceId,
        messageId,
        abortReason: "system",
        metadata: {
          duration: Date.now() - startTime,
        },
      };
      this.emit("stream-abort", abortEvent);

      this.activeQueries.delete(workspaceId);
      return Err({ type: "unknown", raw: `SDK execution failed: ${errorMessage}` });
    }
  }

  /**
   * Handle a partial streaming message (content_block_delta / content_block_start).
   */
  private handleStreamEvent(
    partialMsg: SdkPartialAssistantMessage,
    workspaceId: string,
    messageId: string
  ): void {
    const event = partialMsg.event;

    if (event.type === "content_block_delta" && event.delta) {
      const delta = event.delta;
      if (delta.type === "text_delta" && delta.text) {
        const deltaEvent: StreamDeltaEvent = {
          type: "stream-delta",
          workspaceId,
          messageId,
          delta: delta.text,
          tokens: 0, // Token count not available in deltas
          timestamp: Date.now(),
        };
        this.emit("stream-delta", deltaEvent);
      } else if (delta.type === "thinking_delta" && delta.thinking) {
        const reasoningEvent: ReasoningDeltaEvent = {
          type: "reasoning-delta",
          workspaceId,
          messageId,
          delta: delta.thinking,
          tokens: 0,
          timestamp: Date.now(),
        };
        this.emit("reasoning-delta", reasoningEvent);
      }
    } else if (event.type === "content_block_start" && event.content_block) {
      const contentBlock = event.content_block;
      if (contentBlock.type === "tool_use") {
        const toolBlock = contentBlock as ToolUseBlock;
        const toolStartEvent: ToolCallStartEvent = {
          type: "tool-call-start",
          workspaceId,
          messageId,
          toolCallId: toolBlock.id,
          toolName: toolBlock.name,
          args: toolBlock.input,
          tokens: 0,
          timestamp: Date.now(),
        };
        this.emit("tool-call-start", toolStartEvent);
      }
    }
  }

  /**
   * Handle a complete assistant message - extract parts and emit tool-call-end events.
   */
  private handleAssistantMessage(
    assistantMsg: SdkAssistantMessage,
    workspaceId: string,
    messageId: string,
    parts: CompletedMessagePart[]
  ): void {
    const msgParts = convertContentBlocksToParts(assistantMsg.message.content);
    parts.push(...msgParts);

    // Emit tool-call-end for any tool uses
    for (const part of msgParts) {
      if (part.type === "dynamic-tool") {
        const toolEndEvent: ToolCallEndEvent = {
          type: "tool-call-end",
          workspaceId,
          messageId,
          toolCallId: part.toolCallId,
          toolName: part.toolName,
          result: undefined, // SDK handles tool results internally
          timestamp: Date.now(),
        };
        this.emit("tool-call-end", toolEndEvent);
      }
    }
  }

  /**
   * Stop an active SDK query for a workspace.
   */
  async stopStream(workspaceId: string): Promise<Result<void>> {
    const activeQuery = this.activeQueries.get(workspaceId);
    if (!activeQuery) {
      return Ok(undefined);
    }

    try {
      await activeQuery.interrupt();
      activeQuery.close();
      this.activeQueries.delete(workspaceId);
      return Ok(undefined);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log.error("[AgentSdkService] Failed to stop SDK stream", {
        workspaceId,
        error: errorMessage,
      });
      return Err(`Failed to stop SDK stream: ${errorMessage}`);
    }
  }

  /**
   * Check if a workspace has an active SDK query.
   */
  isStreaming(workspaceId: string): boolean {
    return this.activeQueries.has(workspaceId);
  }
}
