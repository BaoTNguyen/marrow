<!-- arteries:start -->
## Arteries Memory

At session start, run `PLUGIN_DATA=1 node /home/bao-tn/Coding/Projects/marrow/.arteries/hooks/arteries-activate.cjs` and include the output as session context.
On each user prompt, pipe the prompt JSON to `PLUGIN_DATA=1 node /home/bao-tn/Coding/Projects/marrow/.arteries/hooks/arteries-observe.cjs` and use any returned `additionalContext` to guide your response.
When compacting, run `ARTERIES_CLI=codex bash /home/bao-tn/Coding/Projects/marrow/.arteries/hooks/hook-compact-packet.sh codex-compact` and preserve the returned continuity packet.
When an assistant response is available from a hook or transcript event, pipe that event to `ARTERIES_CLI=codex bash /home/bao-tn/Coding/Projects/marrow/.arteries/hooks/hook-assistant-observe.sh codex-assistant` so Arteries can extract assistant-discovered project memory.

Arteries observes turns and assistant responses, builds memory, may surface retrieved prompts, and produces compact continuity packets as additional context.
<!-- arteries:end -->
