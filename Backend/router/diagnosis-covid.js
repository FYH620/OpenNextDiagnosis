const upload = require("../utils/file_uploader");
const log4js = require("log4js");
const logger = log4js.getLogger('diagnosisCovidLogger')
const backend = require("../backend/diagnosis_backend")
const globalConfig = require("../config/config");

module.exports = async (ctx,next)=>{
    let err = await upload.single('file')(ctx, next)
                .then(res=>res)
                .catch(err=>err)
    if(err){
        ctx.body = {
            code:1,
            msg : err.message
        };
    }else{
        ctx.body = {
            code:0,
            positive:0,
        };
        
        logger.info("File received. Processing...");
        const filePath = ctx.file.path;
        await backend.calc(ctx.body, filePath);
        logger.info("Cleaning...");
        await backend.clean(filePath);
        logger.info("All work finished.")
    }
};