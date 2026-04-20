# OpenCode Hermes Evolution 安装包

将此压缩包解压到你的 OpenCode 项目根目录即可。

解压后的文件结构：
  .opencode/
    commands/evolve.md        → /evolve 命令
    agents/evolver.md         → evolver 审查 agent
    evolution-drafts/         → 草稿管理目录
    plugins/hermes-evolution.ts → 可选的自动提示 plugin
  EVOLUTION.md                → 进化强度配置

使用方法：
1. 解压到项目根目录
2. 在 OpenCode 中输入 /evolve 触发进化审查
3. 编辑 EVOLUTION.md 调整进化强度（100%/50%/0%）

注意：
- agents/evolver.md 中的 model 字段默认是 anthropic/claude-sonnet-4-20250514
  请根据你的实际配置修改
- plugins/hermes-evolution.ts 需要 Bun 运行环境，如不需要自动提示可删除