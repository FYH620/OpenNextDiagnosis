const child_process = require('child_process');
const path = require("path");
const log4js = require("log4js");

const globalConfig = require("../config/config");
const garbageDelete = require("../utils/garbage_queue");
const logger = log4js.getLogger('maskDetectLogger')


let detectProcess = null;

async function waitForProcessOutput()
{
    let res;
    for await (const chunk of detectProcess.stdout.iterator({ destroyOnReturn: false })) {
        res = chunk.toString();
        break;
    }
    return res;
}

module.exports = {
    startProcess: async () =>
    {
        if(detectProcess)
            return true;

        logger.info("MaskDetect backend process starting...");
        const executePath = path.join(globalConfig.rootPath,"/../MaskDetect/predict.py");
        const workingDir = path.join(globalConfig.rootPath,"/../MaskDetect");

        detectProcess = child_process.spawn('python', [executePath, '--cuda', globalConfig.cuda.toString()],
            {cwd:workingDir, detached:true}
        );

        let res = await waitForProcessOutput();
        if(res.trim() != "init done.")
        {
            logger.error("Fail to start process! " + res);
            detectProcess.kill();
            detectProcess = null;
            return false;
        }
        logger.info("Process Inited.");
        return true;
    },
    stopProcess: async () =>
    {
        if(!detectProcess)
            return;
        detectProcess.kill();
        detectProcess = null
    },
    calc: async (body, filePath, foggyMode, darkMode) =>
    {
        try
        {
            let foggy = foggyMode == 1 ? "1" : "0";
            let dark = darkMode == 1 ? "1" : "0";
            detectProcess.stdin.write(`${filePath} --foggy ${foggy} --dark ${dark}\r\n`);
            let output = await waitForProcessOutput();
    
            //logger.warn("\n" + output);
            const resArr = output.split('\n');
    
            resArr.forEach((line)=>{
                line = line.trim();
                if(line.indexOf("--") != -1 || line.indexOf("==") != -1 || line.length == 0)
                    return;
    
                const data = line.split(' ');
                body.peoples.push({
                    type:data[0],
                    confidence:parseFloat(data[1]),
                    ax:parseInt(data[2]),
                    ay:parseInt(data[3]),
                    bx:parseInt(data[4]),
                    by:parseInt(data[5]),
                });
            });
        }
        catch(err)
        {
            logger.error("Fail at exec in MaskDetect backend");
            logger.error(err.message);
            body.code = 2;
            body.msg = err.message;
        }
    },
    clean: async (filePath) =>
    {
        garbageDelete(filePath);
    }
}