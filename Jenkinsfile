pipeline {
    agent any

    stages {
        stage('Install') {
            steps {
                echo 'Installing..'
                sh 'scripts/install.sh'
            }
        }
        // stage('Build') {
        //     steps {
        //         echo 'Building..'
        //         sh 'scripts/build.sh'
        //     }
        // }
        // stage('Deploy') {
        //     steps {
        //         echo 'Deploying....'
        //         sh 'scripts/deploy.sh'
        //     }
        // }
    }
}
