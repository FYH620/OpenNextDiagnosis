const fs = require("fs");

module.exports = function(filePath) {
    setTimeout(()=>{fs.unlinkSync(filePath)},500);
};