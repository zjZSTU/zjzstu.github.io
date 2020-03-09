pipeline {
    agent any
    // tools {nodejs "node"}
    environment {
        ID_RSA=credentials('06493beb-9552-4109-b3be-6a90a988f9b5')
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