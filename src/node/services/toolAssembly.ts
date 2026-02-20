import type { Tool } from "ai";
import { applyToolPolicy, type ToolPolicy } from "@/common/utils/tools/toolPolicy";

export interface ApplyToolPolicyOptions {
  allTools: Record<string, Tool>;
  extraTools?: Record<string, Tool>;
  effectiveToolPolicy: ToolPolicy | undefined;
}

export async function applyToolPolicyToTools(
  opts: ApplyToolPolicyOptions
): Promise<Record<string, Tool>> {
  const { allTools, extraTools, effectiveToolPolicy } = opts;
  const allToolsWithExtra = extraTools ? { ...allTools, ...extraTools } : allTools;
  return applyToolPolicy(allToolsWithExtra, effectiveToolPolicy);
}
