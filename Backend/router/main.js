module.exports = function (router) {
    router.get('/', require('./welcome'));

    router.post('/api/mask-detect/upload', require('./mask-detect'));
    router.post('/api/diagnosis-covid/upload', require('./diagnosis-covid'));
};