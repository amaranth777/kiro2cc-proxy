// Copyright (c) 2026 Harllan He. Licensed under MIT.

/**
 * 复制文本到剪贴板
 * 安全上下文优先使用 Clipboard API；不可用或被浏览器拒绝时降级为 execCommand
 */
export async function copyToClipboard(text: string): Promise<void> {
  if (window.isSecureContext && typeof navigator.clipboard?.writeText === 'function') {
    try {
      await navigator.clipboard.writeText(text)
      return
    } catch {
      // Clipboard API 可能存在但被浏览器策略拒绝，继续尝试备用方案。
    }
  }

  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.readOnly = true
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)

  try {
    textarea.focus()
    textarea.select()
    const ok = document.execCommand('copy')
    if (!ok) throw new Error('复制失败：浏览器不支持自动复制')
  } catch {
    throw new Error('复制失败：浏览器不支持自动复制')
  } finally {
    document.body.removeChild(textarea)
  }
}
