const upload = require('../utils/file_uploader');
const router = require('koa-router')();

route.post('/user/file', async (ctx,next)=>{
    let err = await upload.single('file')(ctx, next)
                .then(res=>res)
                .catch(err=>err)
    if(err){
        ctx.body = {
            code:0,
            msg : err.message
        }
    }else{
        ctx.body = {
            code:,
            data:ctx.file
        }
    }
});

module.exports = router.routes();