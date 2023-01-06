module.exports = async (ctx,next)=>{
    ctx.body = {
        '/mask-detect/upload': './mask-detect',
        '/diagnosis-covid/upload': './diagnosis-covid'
    };
};