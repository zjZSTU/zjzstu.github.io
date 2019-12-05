pipeline {
    agent any
    tools {nodejs "node"}
    environment {
        ID_RSA=credentials('fa1b8cd3-c422-4ab6-a240-2d2ba8dc5b4a')
    }
    stages {
        stage('Install') {
            steps {
                echo 'Installing..'
                sh 'scripts/install.sh'
            }
        }
        stage('Build') {
            steps {
                echo 'Building..'
                sh 'scripts/build.sh'
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
                sh 'scripts/deploy.sh'
            }
        }
    }
}