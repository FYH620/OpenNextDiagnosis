// 全局资源
const config = require('./config/config');
const logger = require('./utils/logger')();
const BodyParser = require('koa-bodyparser');

// Koa
const Koa = require('koa');
const app = new Koa();

// 初始化目录
const create_dir = require("./utils/create_dir")
create_dir("./public/upload")

// 初始化后端进程
const maskdetect_backend = require("./backend/maskdetect_backend")
const diagnosis_backend = require("./backend/diagnosis_backend")
maskdetect_backend.startProcess()
diagnosis_backend.startProcess()

// 日志
app.use(require('./middleware/logger'));

// 跨域
app.use(require('koa2-cors')());

// body-parser
app.use(BodyParser());

// 路由
const Router = require('koa-router');
const router = new Router();

const routerRegister = require("./router/main");
routerRegister(router);

app.use(router.routes());
app.use(router.allowedMethods());

// 启动
app.listen(config.port, () => {
    logger.info("Server started: http://localhost:" + config.port);
});