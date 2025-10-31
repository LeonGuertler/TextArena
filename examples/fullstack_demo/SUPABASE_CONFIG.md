# Supabase 配置指南

## 问题：邮件验证链接跳转到 localhost:3000

### 原因
Supabase 默认的重定向 URL 是 `localhost:3000`，需要配置为你的实际应用 URL。

## 解决步骤

### 1. 配置 Supabase Site URL 和 Redirect URLs

1. 登录 [Supabase Dashboard](https://app.supabase.com/)
2. 选择你的项目
3. 导航到 **Authentication** → **URL Configuration**
4. 配置以下 URL：

#### Site URL
**本地开发**：
```
http://localhost:8000
```

**生产环境（Render）**：
```
https://your-app-name.onrender.com
```

#### Redirect URLs
添加以下 URL 到允许列表（每行一个）：

**本地开发**：
```
http://localhost:8000
http://localhost:8000/**
```

**生产环境（Render）**：
```
https://your-app-name.onrender.com
https://your-app-name.onrender.com/**
```

⚠️ **重要**：如果同时在本地和 Render 上使用，两组 URL 都要添加！

### 2. 配置邮件模板（可选，推荐）

1. 在 Supabase Dashboard，导航到 **Authentication** → **Email Templates**
2. 选择 **Confirm signup** 模板
3. 确认模板中的 `{{ .ConfirmationURL }}` 正确
4. 如果需要，可以自定义邮件内容

默认模板：
```html
<h2>Confirm your signup</h2>

<p>Follow this link to confirm your user:</p>
<p><a href="{{ .ConfirmationURL }}">Confirm your mail</a></p>
```

### 3. 前端代码已更新

现在前端注册时会自动使用当前域名作为重定向 URL：

```javascript
const signup = await supabaseClient.auth.signUp({
  email,
  password,
  options: {
    emailRedirectTo: window.location.origin
  }
});
```

### 4. 测试验证流程

1. 注册新账号
2. 检查邮箱
3. 点击验证链接
4. 应该正确跳转到你的应用首页（而不是 localhost:3000）

## 其他认证设置

### 禁用邮箱确认（仅用于开发测试）

如果你想在开发时跳过邮箱验证：

1. 导航到 **Authentication** → **Settings**
2. 找到 **Enable email confirmations**
3. 关闭此选项

⚠️ **警告**：生产环境务必启用邮箱确认！

### 邮箱重发限制

Supabase 有速率限制，防止垃圾邮件：
- 同一邮箱每小时只能发送 3-4 封验证邮件
- 如果超限，等待一小时后再试

## 完整的环境变量清单

确保在 Render Dashboard 中设置了所有环境变量：

```
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
SUPABASE_ANON_KEY=eyJ...
SUPABASE_JWT_SECRET=your-jwt-secret
OPENAI_API_KEY=sk-...
```

## 常见问题

### Q: 邮件没收到？
**A**: 
1. 检查垃圾邮件文件夹
2. 确认 Supabase 项目启用了邮件发送
3. 检查是否超出速率限制

### Q: 验证后仍需要重新登录？
**A**: 这是正常的。点击验证链接后：
1. 邮箱被标记为已验证
2. 返回登录页面
3. 用户需要手动输入密码登录

### Q: 想要验证后自动登录？
**A**: 需要：
1. 在 Redirect URLs 中配置完整路径（如 `/mode1.html`）
2. 在注册时指定：`emailRedirectTo: window.location.origin + '/mode1.html'`
3. 在目标页面处理 URL 中的 token

## 本地 vs 生产环境配置对比

| 配置项 | 本地开发 | Render 生产 |
|--------|---------|------------|
| Site URL | `http://localhost:8000` | `https://your-app.onrender.com` |
| Redirect URLs | `http://localhost:8000/**` | `https://your-app.onrender.com/**` |
| 邮箱确认 | 可选（建议关闭） | 必需（必须开启） |

## 部署后检查清单

- [ ] Supabase Site URL 已设置为 Render URL
- [ ] Redirect URLs 包含 Render URL
- [ ] 所有环境变量已在 Render 中配置
- [ ] 邮箱确认功能已启用
- [ ] 注册新账号测试验证流程
- [ ] 验证邮件链接正确跳转

完成这些设置后，邮件验证应该能正常工作！🎉

