module.exports = {
  apps : [{
	name: 'NextDiagnosis-Backend',
    script: 'app.js',
    watch: true,
	// 不用监听的文件
	ignore_watch: [
	  'node_modules',
	  'logs',
	  'public'
	],
    min_uptime: '10s',
    max_restarts: 5,
  }],
  /*deploy : {
    production : {
      user : 'SSH_USERNAME',
      host : 'SSH_HOSTMACHINE',
      ref  : 'origin/master',
      repo : 'GIT_REPOSITORY',
      path : 'DESTINATION_PATH',
      'pre-deploy-local': '',
      'post-deploy' : 'npm install && pm2 reload ecosystem.config.js --env production',
      'pre-setup': ''
    }
  }*/
};
