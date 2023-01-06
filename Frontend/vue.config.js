const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  transpileDependencies: [
    'vuetify'
  ],
  lintOnSave: false,

  //title
  chainWebpack: config => {
    config
      .plugin('html')
      .tap(args => {
        args[0].title = 'NextDiagnosis';
        return args;
      })
  },
  //axios
  devServer: {
    proxy: {
      '/api': {
        target: 'http://localhost:6006/',
        changeOrigin: true,
        pathRewrite: {
          '^/api': '/api'
        }
      }
    }
  }
})
