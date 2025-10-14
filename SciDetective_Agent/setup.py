#!/usr/bin/env python3
"""
SciDetective Agent Setup and Installation Script

This script helps set up the SciDetective Agent environment and dependencies.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is too old. Requires Python 3.8+")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")

    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False

    # Install Python packages
    success = run_command("pip install -r requirements.txt", "Installing Python packages")
    if not success:
        return False

    # Install spaCy model
    success = run_command("python -m spacy download en_core_web_sm", "Installing spaCy English model")
    return success

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = ["outputs", "demo_outputs", "papers", "visualizations"]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    return True

def setup_environment():
    """Set up environment configuration"""
    print("\n🔧 Setting up environment...")

    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("✅ Created .env file from .env.example")
            print("📝 Please edit .env file to add your API keys if needed")
        else:
            print("⚠️  No .env.example found, skipping environment setup")
    else:
        print("✅ .env file already exists")

    return True

def test_installation():
    """Test the installation"""
    print("\n🧪 Testing installation...")

    try:
        # Test imports
        print("🔄 Testing module imports...")
        sys.path.insert(0, os.path.join(os.getcwd(), 'agent'))

        from agent import SciDetectiveAgent
        print("✅ SciDetective Agent import successful")

        # Initialize agent
        agent = SciDetectiveAgent()
        print("✅ Agent initialization successful")

        # Test basic functionality
        help_result = agent.get_help()
        if help_result.success:
            print("✅ Agent functionality test passed")
            return True
        else:
            print("❌ Agent functionality test failed")
            return False

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main setup function"""
    print("🔬 SciDetective Agent Setup")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        sys.exit(1)

    # Install dependencies
    if not install_dependencies():
        print("\n❌ Setup failed: Could not install dependencies")
        sys.exit(1)

    # Create directories
    if not create_directories():
        print("\n❌ Setup failed: Could not create directories")
        sys.exit(1)

    # Setup environment
    if not setup_environment():
        print("\n❌ Setup failed: Could not setup environment")
        sys.exit(1)

    # Test installation
    if not test_installation():
        print("\n⚠️  Setup completed but tests failed")
        print("You may need to check your environment configuration")
    else:
        print("\n🎉 Setup completed successfully!")

    print("\n🚀 Next steps:")
    print("1. Edit .env file to add API keys (optional)")
    print("2. Run demo: python demo.py")
    print("3. Start MCP server: cd agent && python server.py")
    print("4. Check README.md for detailed usage instructions")

if __name__ == "__main__":
    main()