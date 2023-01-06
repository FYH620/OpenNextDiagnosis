let path = require('path');

let baseLogPath = path.resolve(__dirname, '../logs');
let errorLogPath = path.resolve(__dirname, "../logs/error/error");
let responseLogPath = path.resolve(__dirname, "../logs/response/response");
let commonLogPath = path.resolve(__dirname, "../logs/log");

module.exports = {
    appenders:
    {
        out: { type: 'stdout' },
        error:{
            type: "dateFile",
            filename: errorLogPath,
            alwaysIncludePattern:true,
            pattern: "yyyy-MM-dd.log",
            compress: true,
        },
        response:{
            type: "dateFile",
            filename: responseLogPath,
            alwaysIncludePattern:true,
            pattern: "yyyy-MM-dd.log",
            compress: true,
        },
        common:{
            type: "dateFile",
            filename: commonLogPath,
            alwaysIncludePattern:true,
            pattern: "yyyy-MM-dd.log",
            compress: true,
        },
        common:{
            type: "dateFile",
            filename: commonLogPath,
            alwaysIncludePattern:true,
            pattern: "yyyy-MM-dd.log",
            compress: true,
        },
    },
    categories: {
        default: { appenders: [ 'out' ], level: 'ALL' },
        errorLogger: { appenders: [ 'out', 'error' ], level: 'error' },
        responseLogger: { appenders: [ 'out', 'response' ], level: 'ALL' },
        commonLogger: { appenders: [ 'out', 'common' ], level: 'ALL' },
        diagnosisCovidLogger: { appenders: [ 'out', 'common' ], level: 'ALL' },
        maskDetectLogger: { appenders: [ 'out', 'common' ], level: 'ALL' },
    },
    "baseLogPath": baseLogPath
}