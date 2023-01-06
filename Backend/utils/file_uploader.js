const multer = require('@koa/multer');
const uploadConfig = require("../config/upload_config");

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, uploadConfig.uploadPath);
    },
    filename: function (req, file, cb)
    {
        let type = file.originalname.split('.')[1];
        cb(null, `${file.fieldname}-${Date.now().toString(16)}.${type}`);
    }
});

module.exports = multer({ storage: storage, limits: uploadConfig.limits });