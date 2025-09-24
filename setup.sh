#!/bin/bash

echo "�� DSLR Settings Finder - Setup Script"
echo "======================================"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js found: $(node --version)"

# Install dependencies
echo "📦 Installing dependencies..."
npm install

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created"
    echo "⚠️  Please edit .env and add your OpenAI API key"
else
    echo "✅ .env file already exists"
fi

# Check for API key
if grep -q "your-openai-api-key-here" .env; then
    echo "⚠️  OpenAI API key not configured in .env"
    echo "   Get your API key from: https://platform.openai.com/api-keys"
    echo "   Then edit .env and replace 'your-openai-api-key-here' with your actual key"
else
    echo "✅ OpenAI API key appears to be configured"
fi

echo ""
echo "🚀 Setup complete! Next steps:"
echo "1. Configure your OpenAI API key in .env (if not done already)"
echo "2. Run 'npm start' to start the development server"
echo "3. Open http://localhost:3000 in your browser"
echo ""
echo "📚 For more information, see README.md"
