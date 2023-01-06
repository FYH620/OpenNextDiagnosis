const path = require('path');
const globalConfig = require('./config')

module.exports = {
    port: 3001,
    uploadPath: path.join(globalConfig.staticPath + '/upload'),
    limits: {
        fields: 10, //非文件字段的数量
        fileSize: 20 * 1024 * 1024, //文件大小 (字节)
        files: 1 //文件数量
    }
}
