const child_process = require('child_process');
const path = require("path");
const log4js = require("log4js");

const globalConfig = require("../config/config");
const garbageDelete = require("../utils/garbage_queue");
const logger = log4js.getLogger('diagnosisCovidLogger')


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

        logger.info("DiagnosisCOVID backend process starting...");
        const executePath = path.join(globalConfig.rootPath,"/../DiagnosisCOVID/predict.py");
        const workingDir = path.join(globalConfig.rootPath,"/../DiagnosisCOVID");

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
    calc: async (body, filePath) =>
    {
        try
        {
            detectProcess.stdin.write(filePath + "\r\n");
            let output = await waitForProcessOutput();

            logger.warn("\n" + output);
        
            if(output.indexOf("positive") != -1)
            {
                body.positive = 1;
            }
        }
        catch(err)
        {
            logger.error("Fail at exec in Diagnosis-COVID backend");
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