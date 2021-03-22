#!groovy

node('executor') {
  checkout scm

  def authorName  = sh(returnStdout: true, script: 'git --no-pager show --format="%an" --no-patch')
  def allProcessors = sh(returnStdout: true, script: 'cat docker-compose*yml | grep "_processor:" | sed \'s/\\://g\' | sed \'s/_/-/g\' | sort | uniq | xargs').split()
  def isMain    = env.BRANCH_NAME == "main"
  def serviceName = env.JOB_NAME.tokenize("/")[1]

  def commitHash  = sh(returnStdout: true, script: 'git rev-parse HEAD | cut -c-7').trim()
  def imageTag    = "${env.BUILD_NUMBER}-${commitHash}"

  try {
    node('dev-executor') {
      ws("svs-processor-testing-${UUID.randomUUID().toString()}") {
        checkout scm
        stage("Test") {
          sh "make clean"
          sh "make test"
        }
      }
    }

    if(isMain) {
      stage("Build Image") {
        sh "IMAGE_TAG=${imageTag} make build"
        sh "IMAGE_TAG=latest make build"
      }

      stage("Push Image") {
        sh "IMAGE_TAG=${imageTag} make push"
        sh "IMAGE_TAG=latest make push"
      }

      stage("Deploy") {
        allProcessors.each { proc ->
          build job: "service-deploy/pennsieve-non-prod/us-east-1/dev-vpc-use1/dev/${proc}",
          parameters: [
            string(name: 'IMAGE_TAG', value: imageTag),
            string(name: 'TERRAFORM_ACTION', value: 'apply')
          ]
        }
      }

    }
  } catch (e) {
    slackSend(color: '#b20000', message: "FAILED: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL}) by ${authorName}")
    throw e
  }

  slackSend(color: '#006600', message: "SUCCESSFUL: Job '${env.JOB_NAME} [${env.BUILD_NUMBER}]' (${env.BUILD_URL}) by ${authorName}")
}
