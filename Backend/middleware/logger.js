const logger = require('../utils/logger')();

module.exports = async (ctx, next) => {
    const start = new Date();
    try {
      await next();
      let ms = new Date() - start;
      logger.logResponse(ctx, ms);
  
    } catch (error) {
      let ms = new Date() - start;
      logger.logError(ctx, error, ms);
    }
};