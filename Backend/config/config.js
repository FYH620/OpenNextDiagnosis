const path = require('path');

module.exports = {
    port: 6006,
    staticPath: path.join(__dirname + '/../public'),
    rootPath: path.join(__dirname + '/..'),
    cuda: 0,
}