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
                // Use 'bat' for Windows agents, or 'sh' for Linux/macOS agents.
                // Assuming a Windows environment since your OS is Windows, but change 'bat' to 'sh' if Jenkins runs on Linux.
                bat '''
                    python -m venv venv
                    call venv\\Scripts\\activate
                    python -m pip install --upgrade pip
                    pip install -r requirements.txt
                '''
            }
        }

        stage('Syntax Check / Lint') {
            steps {
                // Since there are no unit tests yet, we can do a basic syntax check to ensure the code compiles.
                bat '''
                    call venv\\Scripts\\activate
                    python -m py_compile web_app.py main.py src/*.py
                '''
            }
        }
        
        // You can add a 'Test' stage here in the future if you add pytest or unittest files!
        // stage('Test') { ... }
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
