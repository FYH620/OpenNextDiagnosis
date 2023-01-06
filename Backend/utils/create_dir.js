let fs = require('fs');

module.exports = function(pathStr) {
    try
    {
        if (!fs.existsSync(pathStr)) {
            fs.mkdirSync(pathStr, { recursive: true });
            //console.log('createPath: ' + pathStr);
        }
    }
    catch(e)
    {
        ;
    }
}