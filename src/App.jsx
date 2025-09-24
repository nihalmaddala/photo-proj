import React, { useState, useRef, useEffect } from 'react';
import apiService from './services/apiService';

const LoadingSpinner = () => (
  <div className="flex items-center justify-center">
    <div className="relative">
      <div className="w-12 h-12 border-4 border-gray-200 rounded-full animate-pulse"></div>
      <div className="absolute inset-0 w-12 h-12 border-4 border-accent border-t-transparent rounded-full animate-spin"></div>
    </div>
  </div>
);

const SettingCard = ({ title, value, explanation, isUpdating }) => (
  <div className={`bg-white rounded-xl shadow-lg p-6 transition-all duration-500 ${isUpdating ? 'scale-105 ring-2 ring-accent' : ''}`}>
    <div className="text-center">
      <h3 className="text-lg font-semibold text-gray-800 mb-2">{title}</h3>
      <div className={`text-3xl font-bold text-accent mb-3 transition-all duration-300 ${isUpdating ? 'animate-pulse' : ''}`}>
        {value}
      </div>
      <p className="text-sm text-gray-600 leading-relaxed">{explanation}</p>
    </div>
  </div>
);

const ChatMessage = ({ message, isUser = false, timestamp }) => (
  <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
    <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
      isUser 
        ? 'bg-accent text-white' 
        : 'bg-gray-100 text-gray-800'
    }`}>
      <p className="text-sm">{message}</p>
      {timestamp && (
        <p className={`text-xs mt-1 ${isUser ? 'text-blue-100' : 'text-gray-500'}`}>
          {timestamp}
        </p>
      )}
    </div>
  </div>
);

const App = () => {
  const [currentStep, setCurrentStep] = useState('landing'); // landing, analyzing, results, refining
  const [uploadedImage, setUploadedImage] = useState(null);
  const [settings, setSettings] = useState(null);
  const [userInput, setUserInput] = useState('');
  const [isRefining, setIsRefining] = useState(false);
  const [updatingCards, setUpdatingCards] = useState(false);
  const [chatHistory, setChatHistory] = useState([]);
  const fileInputRef = useRef(null);
  const chatEndRef = useRef(null);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory]);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid image file.');
      return;
    }

    // Create image URL for display
    const imageUrl = URL.createObjectURL(file);
    setUploadedImage(imageUrl);

    setCurrentStep('analyzing');

    try {
      const analysisResult = await apiService.analyzeImage(file);
      setSettings(analysisResult);
      
      // Add initial analysis message to chat
      const analysisMessage = `I've analyzed your photo! Based on the lighting and subject, here are the recommended DSLR settings:`;
      setChatHistory([{
        message: analysisMessage,
        isUser: false,
        timestamp: new Date().toLocaleTimeString()
      }]);
      
      setCurrentStep('results');
    } catch (error) {
      console.error('Analysis failed:', error);
      alert(`Sorry, we encountered an error analyzing your image: ${error.message}`);
      setCurrentStep('landing');
    }
  };

  const handleRefinement = async () => {
    if (!userInput.trim()) return;

    // Add user message to chat
    const userMessage = {
      message: userInput,
      isUser: true,
      timestamp: new Date().toLocaleTimeString()
    };
    setChatHistory(prev => [...prev, userMessage]);

    setIsRefining(true);
    setUpdatingCards(true);

    try {
      const refinedSettings = await apiService.refineSettings(userInput, settings);
      
      // Add AI response to chat
      const aiResponse = `Great idea! ${refinedSettings.reasoning || 'I\'ve adjusted the settings to achieve your vision.'} Here are the updated settings:`;
      const aiMessage = {
        message: aiResponse,
        isUser: false,
        timestamp: new Date().toLocaleTimeString()
      };
      
      // Update settings with animation
      setTimeout(() => {
        setSettings(refinedSettings);
        setUpdatingCards(false);
        setUserInput('');
        setIsRefining(false);
        setChatHistory(prev => [...prev, aiMessage]);
      }, 500);
      
    } catch (error) {
      console.error('Refinement failed:', error);
      setIsRefining(false);
      setUpdatingCards(false);
      
      // Add error message to chat
      const errorMessage = {
        message: `Sorry, I couldn't process that request. Please try again with a different description.`,
        isUser: false,
        timestamp: new Date().toLocaleTimeString()
      };
      setChatHistory(prev => [...prev, errorMessage]);
    }
  };

  const resetApp = () => {
    setCurrentStep('landing');
    setUploadedImage(null);
    setSettings(null);
    setUserInput('');
    setIsRefining(false);
    setUpdatingCards(false);
    setChatHistory([]);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  // Landing Page
  if (currentStep === 'landing') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 font-inter">
        {/* Hero Background Image */}
        <div 
          className="absolute inset-0 bg-cover bg-center bg-no-repeat opacity-20"
          style={{
            backgroundImage: "url('https://images.unsplash.com/photo-1606983340126-99ab4feaa64a?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2574&q=80')"
          }}
        />
        
        {/* Main Content */}
        <div className="relative z-10 min-h-screen flex items-center justify-center px-6">
          <div className="text-center max-w-2xl">
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-8 leading-tight">
              Translate Your Phone's Photo into 
              <span className="text-accent block mt-2">DSLR Settings</span>
            </h1>
            
            <p className="text-xl text-gray-600 mb-12 leading-relaxed">
              Upload any photo and discover the perfect camera settings to recreate that shot.
            </p>
            
            <button
              onClick={() => fileInputRef.current?.click()}
              className="inline-flex items-center justify-center px-12 py-6 bg-accent hover:bg-accent-hover text-white text-xl font-semibold rounded-2xl shadow-xl hover:shadow-2xl transform hover:scale-105 transition-all duration-200 focus:outline-none focus:ring-4 focus:ring-accent/25"
            >
              Upload a Photo to Start
            </button>
            
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="hidden"
            />
          </div>
        </div>
      </div>
    );
  }

  // Analysis Loading Page
  if (currentStep === 'analyzing') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 font-inter flex items-center justify-center px-6">
        <div className="text-center max-w-md">
          <div className="mb-8">
            <LoadingSpinner />
          </div>
          <h2 className="text-2xl font-semibold text-gray-900 mb-4">
            Analyzing Your Photo
          </h2>
          <p className="text-gray-600">
            Examining lighting conditions, subject matter, and composition to determine optimal DSLR settings...
          </p>
        </div>
      </div>
    );
  }

  // Results Page with Chat Interface
  if (currentStep === 'results') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-gray-100 font-inter py-8 px-6">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              Your DSLR Settings
            </h2>
            <p className="text-gray-600">
              Adjust these settings on your camera to recreate this shot
            </p>
          </div>

          {/* Image and Settings Layout */}
          <div className="grid lg:grid-cols-2 gap-8 mb-8">
            {/* Uploaded Image */}
            <div className="order-2 lg:order-1">
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h3 className="text-xl font-semibold text-gray-900 mb-4">Your Photo</h3>
                <div className="relative">
                  <img
                    src={uploadedImage}
                    alt="Uploaded photo for analysis"
                    className="w-full h-auto rounded-xl shadow-md max-h-96 object-cover"
                  />
                  <div className="absolute top-4 right-4 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-sm font-medium text-gray-700">
                    Reference
                  </div>
                </div>
              </div>
            </div>

            {/* Settings Cards */}
            <div className="order-1 lg:order-2 space-y-6">
              <SettingCard
                title="ISO"
                value={settings.iso}
                explanation={settings.explanations.iso}
                isUpdating={updatingCards}
              />
              <SettingCard
                title="Aperture"
                value={settings.aperture}
                explanation={settings.explanations.aperture}
                isUpdating={updatingCards}
              />
              <SettingCard
                title="Shutter Speed"
                value={settings.shutterSpeed}
                explanation={settings.explanations.shutterSpeed}
                isUpdating={updatingCards}
              />
            </div>
          </div>

          {/* Chat Interface */}
          <div className="bg-white rounded-2xl shadow-lg p-6 mb-8">
            <h3 className="text-xl font-semibold text-gray-900 mb-4">
              Want to adjust these settings?
            </h3>
            
            {/* Chat Messages */}
            <div className="h-64 overflow-y-auto mb-4 border rounded-lg p-4 bg-gray-50">
              {chatHistory.map((msg, index) => (
                <ChatMessage
                  key={index}
                  message={msg.message}
                  isUser={msg.isUser}
                  timestamp={msg.timestamp}
                />
              ))}
              {isRefining && (
                <div className="flex justify-start mb-4">
                  <div className="bg-gray-100 text-gray-800 px-4 py-3 rounded-2xl">
                    <div className="flex items-center">
                      <LoadingSpinner />
                      <span className="ml-2 text-sm">Thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
            
            {/* Chat Input */}
            <div className="flex gap-3">
              <input
                type="text"
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                placeholder="e.g., 'Make the background blurry' or 'Freeze the motion'"
                className="flex-1 px-4 py-3 border-2 border-gray-200 rounded-xl focus:border-accent focus:outline-none"
                onKeyPress={(e) => e.key === 'Enter' && !isRefining && handleRefinement()}
                disabled={isRefining}
              />
              <button
                onClick={handleRefinement}
                disabled={isRefining || !userInput.trim()}
                className="px-6 py-3 bg-accent hover:bg-accent-hover text-white font-semibold rounded-xl disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
              >
                {isRefining ? 'Sending...' : 'Send'}
              </button>
            </div>

            <div className="mt-3 text-center text-sm text-gray-500">
              Try: "blur background", "freeze motion", "everything sharp", "dreamy effect"
            </div>
          </div>

          {/* Action Buttons */}
          <div className="text-center space-x-4">
            <button
              onClick={() => fileInputRef.current?.click()}
              className="px-8 py-3 bg-gray-200 hover:bg-gray-300 text-gray-800 font-semibold rounded-xl transition-all duration-200"
            >
              Try Another Photo
            </button>
            <button
              onClick={resetApp}
              className="px-8 py-3 bg-accent hover:bg-accent-hover text-white font-semibold rounded-xl transition-all duration-200"
            >
              Start Over
            </button>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
          />
        </div>
      </div>
    );
  }

  return null;
};

export default App;
