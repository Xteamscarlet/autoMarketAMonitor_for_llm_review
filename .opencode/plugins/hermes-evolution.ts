import type { Plugin } from "@opencode-ai/plugin"

/**
 * Hermes Evolution Plugin for OpenCode
 *
 * Automatically prompts the user to run /evolve when a session goes idle
 * after significant activity. This is the "automatic trigger" complement
 * to the manual /evolve command.
 *
 * Behavior:
 * - Tracks tool call count per session as a proxy for "work done"
 * - When session.idle fires and tool calls >= 5, shows a toast suggesting /evolve
 * - Only prompts once per session to avoid nagging
 * - Respects EVOLUTION.md intensity setting (0% = never prompt)
 */

const EVOLVE_THRESHOLD = 5 // minimum tool calls before suggesting evolution

export const HermesEvolution: Plugin = async ({ client, $ }) => {
  let toolCallCount = 0
  let prompted = false

  // Read evolution intensity from EVOLUTION.md
  async function getIntensity(): Promise<number> {
    try {
      const content = await $`cat EVOLUTION.md 2>/dev/null || echo "100%"`
      const text = String(content)
      const match = text.match(/\*\*Current:\s*(\d+)%/)
      return match ? parseInt(match[1]) : 100
    } catch {
      return 100
    }
  }

  return {
    // Track tool usage as a proxy for session activity
    "tool.execute.after": async () => {
      toolCallCount++
    },

    // When session goes idle, suggest evolution if enough activity
    event: async ({ event }) => {
      if (event.type === "session.idle" && !prompted) {
        const intensity = await getIntensity()
        if (intensity === 0) return

        if (toolCallCount >= EVOLVE_THRESHOLD) {
          prompted = true
          // Show a toast suggesting the user run /evolve
          // The user can ignore this or run /evolve manually
          await client.app.log({
            body: {
              service: "hermes-evolution",
              level: "info",
              message: `Session had ${toolCallCount} tool calls. Consider running /evolve to review for learnings.`,
            },
          })
        }
      }
    },
  }
}