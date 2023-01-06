const log4js = require('log4js');
const logConfig = require('../config/log_config');
const createDir = require('./create_dir');
const path = require('path')

module.exports = function()
{
    //Init
    if(logConfig.baseLogPath)
        createDir(logConfig.baseLogPath);
    for (var i in logConfig.appenders)
    {
        if (logConfig.appenders[i].path)
            createDir(path.dirname(logConfig.appenders[i].path));
    }
    log4js.configure(logConfig);

    //Logger
    const logger = log4js.getLogger('commonLogger');
    logger.logError = function (ctx, error, responseTime) {
        if (ctx && error) {
            log4js.getLogger('errorLogger').error(formatError(ctx, error, responseTime));
        }
    };
    logger.logResponse = function (ctx, responseTime) {
        if (ctx) {
            log4js.getLogger('responseLogger').info(formatResponse(ctx, responseTime));
        }
    };
    return logger;
};

//格式化响应日志
function formatResponse(ctx, resTime) {
    let logText = new String();

    logText += "\n*************** Response Log Begin ***************" + "\n";
    logText += formatReqLog(ctx.request, resTime);
    logText += "Response Status: " + ctx.status + "\n";
    //logText += "Response Body: " + "\n" + JSON.stringify(ctx.body) + "\n";
    logText += "*************** Response Log End ***************" + "\n";

    return logText;

}

//格式化错误日志
function formatError(ctx, err, resTime) {
    let logText = new String();

    logText += "\n*************** Error Log Begin ***************" + "\n";
    logText += formatReqLog(ctx.request, resTime);
    logText += "Error Name: " + err.name + "\n";
    logText += "Error Message: " + err.message + "\n";
    logText += "Error Stack: " + err.stack + "\n";
    logText += "*************** Error Log End ***************" + "\n";

    return logText;
};

//格式化请求日志
function formatReqLog(req, resTime) {

    let logText = new String();

    let method = req.method;
    logText += "Request Method: " + method + "\n";
    logText += "Request OriginalUrl:  " + req.originalUrl + "\n";
    logText += "Request Client IP:  " + req.ip + "\n";

    let startTime;
    if (method === 'GET')
        logText += "Request Query:  " + JSON.stringify(req.query) + "\n";
    //else
        //logText += "Request Body: " + "\n" + JSON.stringify(req.body) + "\n";
    logText += "Response Time: " + resTime + "\n";

    return logText;
}
