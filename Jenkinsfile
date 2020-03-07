pipeline {
    agent any
    tools {nodejs "node"}
    environment {
        ID_RSA=credentials('e46b6d92-f84d-491c-93f4-f0c1055e87f9')
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