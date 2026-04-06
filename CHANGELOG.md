# Changelog

## v4.2.26

- 将 LLM 触发的自拍参考图请求改成后台任务模式
- `aiimg_generate` 在 explicit / auto / follow-up 三种自拍入口下都会立即返回 accepted，不再同步阻塞等待 100 秒以上
- 生成结果仍会自动发给用户，失败时也会自动写回对话结果并做失败标记

## v4.2.25

- 修复旧 LLM 工具 `gitee_draw_image` 在自拍语义下仍强制走文生图的问题
- 现在如果提示词明显是在请求 bot 自拍或你自己的自拍，会自动切到 `selfie_ref` 自拍参考图链路
- 普通绘图请求仍保持原来的文生图行为
