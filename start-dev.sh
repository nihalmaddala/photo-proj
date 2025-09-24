#!/bin/bash

echo "🚀 Starting DSLR Settings Finder Development Environment"
echo "========================================================"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 16+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js found: $(node --version)"

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd /Users/nihalmaddala/photo-proj
npm install

if [ $? -eq 0 ]; then
    echo "✅ Frontend dependencies installed"
else
    echo "❌ Failed to install frontend dependencies"
    exit 1
fi

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd backend
npm install

if [ $? -eq 0 ]; then
    echo "✅ Backend dependencies installed"
else
    echo "❌ Failed to install backend dependencies"
    exit 1
fi

# Check for backend .env file
if [ ! -f ".env" ]; then
    echo "📝 Creating backend .env file..."
    cp .env.example .env
    echo "✅ Backend .env file created"
    echo "⚠️  Please edit backend/.env and add your OpenAI API key"
else
    echo "✅ Backend .env file already exists"
fi

# Check for API key
if grep -q "your-openai-api-key-here" .env; then
    echo "⚠️  OpenAI API key not configured in backend/.env"
    echo "   Get your API key from: https://platform.openai.com/api-keys"
    echo "   Then edit backend/.env and replace 'your-openai-api-key-here' with your actual key"
else
    echo "✅ OpenAI API key appears to be configured"
fi

cd ..

echo ""
echo "🚀 Starting both frontend and backend servers..."
echo ""
echo "📱 Frontend will be available at: http://localhost:3000"
echo "🔧 Backend API will be available at: http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Start backend server in background
cd backend
npm start &
BACKEND_PID=$!

# Start frontend server
cd ..
npm start &
FRONTEND_PID=$!

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Stopping servers..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Servers stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for both processes
wait $BACKEND_PID $FRONTEND_PID
