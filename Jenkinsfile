pipeline {
    agent any

    environment {
        // Optional: define any environment variables needed for the build
        PYTHONUNBUFFERED = '1'
    }

    stages {
        stage('Checkout') {
            steps {
                // Checkout the repository from source control
                checkout scm
            }
        }

        stage('Install Dependencies') {
            steps {
                // Jenkins in Docker runs on Linux, so we use 'sh' and Linux paths
                sh '''
                    python3 -m venv venv
                    . venv/bin/activate
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Syntax Check / Lint') {
            steps {
                // Ensure syntax check runs correctly on Linux
                sh '''
                    . venv/bin/activate
                    python -m py_compile web_app.py main.py src/*.py
                '''
            }
        }
    }

    post {
        always {
            // Clean up the workspace after the run
            cleanWs()
        }
        success {
            echo 'Build completed successfully!'
        }
        failure {
            echo 'Build failed. Please check the logs.'
        }
    }
}
