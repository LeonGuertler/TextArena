# 演示模式 (Demo Mode)

## 概述

应用现在支持两种运行模式：

### 🎮 演示模式（Demo Mode）
- **触发条件**: 未设置 `SUPABASE_JWT_SECRET` 环境变量
- **特点**:
  - 无需 Supabase 账户
  - 无需用户认证/登录
  - 游戏日志输出到控制台（不保存到数据库）
  - 适合快速测试和演示

### 🔒 生产模式（Production Mode）
- **触发条件**: 设置了所有 Supabase 环境变量
- **特点**:
  - 需要 Supabase JWT 认证
  - 游戏结果保存到 Supabase 数据库
  - 完整的用户管理
  - 适合正式部署

## 在 Render 上使用演示模式

如果你想快速部署并测试应用，**不设置** Supabase 相关环境变量即可自动启用演示模式：

### 需要设置的环境变量（演示模式）：
```bash
OPENAI_API_KEY=your_openai_api_key
FULLSTACK_DEMO_RELOAD=0
PYTHON_VERSION=3.10.0
```

### 需要设置的环境变量（生产模式）：
```bash
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_JWT_SECRET=your_jwt_secret
FULLSTACK_DEMO_RELOAD=0
PYTHON_VERSION=3.10.0
```

## 健康检查

访问 `/health` 端点查看当前运行模式：

```bash
curl https://your-app.onrender.com/health
```

响应示例：
```json
{
  "status": "healthy",
  "demo_mode": true,
  "message": "Demo mode: authentication disabled"
}
```

## 演示模式的限制

- 游戏记录不会持久化
- 所有用户使用相同的 "demo-user" ID
- 不支持多用户隔离
- 仅用于测试和演示目的

## 切换到生产模式

1. 创建 Supabase 项目
2. 在 Render Dashboard 添加所有 Supabase 环境变量
3. 重新部署应用
4. 前端需要配置 Supabase 登录功能

