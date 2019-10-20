pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                echo 'Building..'
                pwd
                ls -alh
            }
        }
        stage('Test') {
            steps {
                echo 'Testing..'
                git branch -vv
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}