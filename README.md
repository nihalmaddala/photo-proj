# DSLR Settings Finder

An AI-powered web application that analyzes uploaded photos and provides optimal DSLR camera settings, with intelligent natural language refinement capabilities.

## 🎯 Features

- **AI-Powered Image Analysis**: Uses OpenAI's GPT-4 Vision to analyze photos and determine optimal camera settings
- **Natural Language Refinement**: Conversational AI that understands user intent and adjusts settings accordingly
- **Secure Backend Architecture**: API keys managed server-side, never exposed to users
- **Real-time Settings Display**: Clean, intuitive interface showing ISO, Aperture, and Shutter Speed
- **Professional Photography Guidance**: Contextual tips and explanations for each setting
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🏗️ Architecture

### Backend (Node.js/Express)
- **Secure API Key Management**: OpenAI API keys stored server-side only
- **Image Processing**: Handles file uploads and converts to base64 for AI analysis
- **Rate Limiting**: Prevents API abuse and manages costs
- **Error Handling**: Graceful fallbacks when AI services fail
- **CORS Security**: Proper cross-origin resource sharing configuration

### Frontend (React)
- **Clean API Integration**: Communicates with backend via REST API
- **Progressive Enhancement**: Works with/without AI backend
- **Real-time Status**: Shows backend health and AI availability
- **Responsive UI**: Mobile-first design with Tailwind CSS

### AI Integration Strategy

**Phase 1: OpenAI Integration (Current)**
- Immediate functionality with GPT-4 Vision and GPT-4
- Structured prompts for consistent responses
- Fallback to rule-based system when AI unavailable

**Phase 2: Custom Model Training (Future)**
- Train specialized model on photography datasets
- Fine-tune for DSLR settings prediction
- Reduce API costs and improve response times

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ 
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Automated Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd photo-proj

# Run the automated setup script
./start-dev.sh
```

This will:
- Install all dependencies (frontend + backend)
- Create necessary configuration files
- Start both servers simultaneously
- Open the application at `http://localhost:3000`

### Manual Setup

1. **Install Dependencies**
   ```bash
   # Frontend
   npm install
   
   # Backend
   cd backend
   npm install
   cd ..
   ```

2. **Configure OpenAI API Key**
   ```bash
   # Edit backend/.env
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. **Start Servers**
   ```bash
   # Terminal 1: Backend
   cd backend
   npm start
   
   # Terminal 2: Frontend
   npm start
   ```

4. **Access Application**
   - Frontend: `http://localhost:3000`
   - Backend API: `http://localhost:3001`

## 🔧 Configuration

### Backend Environment Variables

```env
# Required for AI functionality
OPENAI_API_KEY=sk-your-openai-api-key

# Server Configuration
PORT=3001
NODE_ENV=development

# Security
CORS_ORIGIN=http://localhost:3000

# Rate Limiting
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100
```

### Frontend Environment Variables

```env
# Optional: Custom backend URL (defaults to localhost:3001)
REACT_APP_API_URL=http://localhost:3001/api
```

## 📱 Usage

### Basic Workflow

1. **Upload Photo**: Click "Upload a Photo to Start"
2. **AI Analysis**: Wait 3-5 seconds for AI to analyze lighting, subject, composition
3. **Review Settings**: See recommended ISO, Aperture, Shutter Speed with explanations
4. **Refine Settings**: Use natural language to adjust for specific creative goals
5. **Apply Settings**: Use the recommended settings on your DSLR

### Example Refinement Requests

- **"Make everything blurry"** → f/1.4 aperture, fast shutter
- **"Keep everything sharp"** → f/16 aperture, slower shutter  
- **"Freeze motion"** → Fast shutter speed, higher ISO
- **"Smooth water effect"** → Long exposure, small aperture, tripod tip
- **"Dreamy portrait"** → Wide aperture, soft focus

## �� Design Philosophy

### User Experience
- **Simplicity First**: Single linear flow, no complex navigation
- **Instant Value**: Immediate settings recommendations
- **Progressive Enhancement**: Works without AI, better with AI
- **Educational**: Clear explanations help users learn photography

### Technical Design
- **Secure Architecture**: API keys never exposed to frontend
- **Clean Separation**: Backend handles AI, frontend handles UI
- **Error Resilience**: Graceful fallbacks when services fail
- **Performance**: Optimized for fast loading and responsive interactions
- **Scalability**: Easy to extend with additional AI models or features

## 🔮 Future Enhancements

### Short Term
- [ ] Custom model training pipeline
- [ ] Batch photo analysis
- [ ] Settings history and favorites
- [ ] Camera-specific recommendations

### Long Term
- [ ] Real-time camera integration
- [ ] Advanced composition analysis
- [ ] Community settings sharing
- [ ] Mobile app development

## 🛠️ Development

### Project Structure
```
photo-proj/
├── backend/
│   ├── services/
│   │   └── aiService.js      # AI integration layer
│   ├── server.js             # Express server
│   ├── package.json          # Backend dependencies
│   └── .env                  # Backend configuration
├── src/
│   ├── services/
│   │   └── apiService.js     # Frontend API client
│   ├── App.jsx              # Main application component
│   └── index.js             # React entry point
├── package.json             # Frontend dependencies
└── start-dev.sh            # Development startup script
```

### Key Architectural Decisions

1. **Backend-First AI**: All AI operations handled server-side for security
2. **API-First Design**: Clean REST API for frontend-backend communication
3. **Progressive Enhancement**: App works without AI, enhanced with AI
4. **Security by Default**: Rate limiting, CORS, input validation

### Adding New AI Features

1. **Extend Backend**: Add new endpoints to `backend/server.js`
2. **Update AI Service**: Add new methods to `backend/services/aiService.js`
3. **Enhance Frontend**: Add new API calls to `src/services/apiService.js`
4. **Test Fallbacks**: Ensure graceful degradation when AI unavailable

## 📊 Performance Considerations

- **API Costs**: OpenAI API usage optimized with token limits and rate limiting
- **Response Times**: 3-5 second analysis, 2 second refinement
- **Fallback Speed**: Instant responses when AI unavailable
- **Image Processing**: Efficient base64 encoding for API transmission
- **Rate Limiting**: Prevents abuse and manages API costs

## 🔒 Security & Privacy

- **API Key Security**: Keys stored server-side only, never exposed to frontend
- **Input Validation**: File type and size validation on uploads
- **Rate Limiting**: Prevents API abuse and manages costs
- **CORS Protection**: Proper cross-origin resource sharing configuration
- **Error Handling**: No sensitive information leaked in error messages
- **Image Data**: Transmitted to OpenAI for analysis, not stored permanently

## 📈 Monitoring & Analytics

Consider adding:
- AI response quality metrics
- User refinement success rates  
- API usage and cost tracking
- Performance monitoring
- Error rate monitoring

## 🚀 Deployment

### Development
```bash
./start-dev.sh
```

### Production Considerations
- Deploy backend to secure server (AWS, Google Cloud, etc.)
- Use environment variables for production API keys
- Set up proper CORS origins for production domains
- Configure rate limiting for production traffic
- Set up monitoring and logging

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 Vision and GPT-4 APIs
- React and Express communities
- Photography experts who provided domain knowledge
- Beta testers who helped refine the user experience
