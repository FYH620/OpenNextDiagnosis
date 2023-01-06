const util = require("util");
const fs = require("fs");
const path = require("path");

const globalConfig = require("../config/config");
const log4js = require("log4js");
const logger = log4js.getLogger('maskDetectLogger')
const backend = require("../backend/maskdetect_backend")

async function saveImage(image,filePath)
{
    try
    {
        let imageBitmap = Buffer.from(image, 'base64'); // 解码图片
        await util.promisify(fs.writeFile)(filePath, imageBitmap);
    }
    catch(err)
    {
        logger.error("Fail at saveImage in MaskDetect");
        logger.error(err.message);
    }
}

module.exports = async (ctx,next)=>
{
    logger.info("Image received. Processing...");
    const filePath = path.join(globalConfig.staticPath
        ,`/upload/MaskSnapshot-${Date.now().toString(16)}.png`);
    
    //logger.warn("\n" + filePath);
    await saveImage(ctx.request.body.image, filePath);

    ctx.body = {
        code:0,
        peoples:[]
    };
    await backend.calc(ctx.body, filePath, ctx.request.body.foggyMode, ctx.request.body.darkMode);
    logger.info("Cleaning...");
    await backend.clean(filePath);
    logger.info("All work finished.");
};