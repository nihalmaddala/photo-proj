const OpenAI = require('openai');

// Initialize OpenAI client with server-side API key
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// Photography Context Prompts
const PHOTOGRAPHY_PROMPTS = {
  IMAGE_ANALYSIS: `
You are a professional photography expert analyzing an image to determine optimal DSLR camera settings.

Analyze the uploaded image and provide camera settings based on:
1. Lighting conditions (bright, dim, artificial, natural)
2. Subject type (portrait, landscape, action, macro, etc.)
3. Desired depth of field
4. Motion requirements
5. Image quality priorities

Respond with ONLY a JSON object in this exact format:
{
  "iso": 100,
  "aperture": "f/4.0",
  "shutterSpeed": "1/250s",
  "explanations": {
    "iso": "Brief explanation of ISO choice",
    "aperture": "Brief explanation of aperture choice", 
    "shutterSpeed": "Brief explanation of shutter speed choice"
  },
  "photographyType": "portrait|landscape|action|macro|street|night|other",
  "lightingCondition": "bright|dim|artificial|mixed",
  "confidence": 0.85
}
`,

  REFINEMENT_ANALYSIS: `
You are a professional photography expert helping refine DSLR camera settings based on user intent.

Current settings: ISO {currentISO}, Aperture {currentAperture}, Shutter Speed {currentShutterSpeed}

User request: "{userInput}"

Analyze the user's request and determine new camera settings that achieve their creative vision.
Consider:
- Depth of field requirements
- Motion blur preferences  
- Lighting adjustments
- Subject isolation needs
- Artistic effects desired

Respond with ONLY a JSON object in this exact format:
{
  "iso": 100,
  "aperture": "f/2.8", 
  "shutterSpeed": "1/500s",
  "explanations": {
    "iso": "Brief explanation of ISO adjustment",
    "aperture": "Brief explanation of aperture adjustment",
    "shutterSpeed": "Brief explanation of shutter speed adjustment"
  },
  "reasoning": "Conversational explanation of why these settings are better for achieving the user's goal. Be friendly and educational, like talking to a friend.",
  "tip": "Optional pro tip for the user (null if none)",
  "confidence": 0.90
}
`
};

// Camera Settings Ranges (for validation and suggestions)
const CAMERA_SETTINGS = {
  ISO: {
    MIN: 50,
    MAX: 12800,
    COMMON_VALUES: [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
  },
  APERTURE: {
    MIN: 1.0,
    MAX: 32.0,
    COMMON_VALUES: [1.4, 1.8, 2.0, 2.8, 4.0, 5.6, 8.0, 11.0, 16.0, 22.0]
  },
  SHUTTER_SPEED: {
    MIN: 1/8000, // 1/8000s
    MAX: 30, // 30 seconds
    COMMON_VALUES: ['1/8000', '1/4000', '1/2000', '1/1000', '1/500', '1/250', '1/125', '1/60', '1/30', '1/15', '1/8', '1/4', '1/2', '1"', '2"', '4"', '8"', '15"', '30"']
  }
};

class AIService {
  constructor() {
    this.isConfiguredFlag = !!process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY !== 'your-openai-api-key-here';
  }

  /**
   * Analyze uploaded image and provide DSLR settings
   * @param {Object} imageFile - The uploaded image file from multer
   * @returns {Promise<Object>} Camera settings and analysis
   */
  async analyzeImage(imageFile) {
    if (!this.isConfiguredFlag) {
      return this.getFallbackAnalysis();
    }

    try {
      // Convert buffer to base64 for OpenAI Vision API
      const base64Image = imageFile.buffer.toString('base64');
      
      const response = await openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: "user",
            content: [
              {
                type: "text",
                text: PHOTOGRAPHY_PROMPTS.IMAGE_ANALYSIS
              },
              {
                type: "image_url",
                image_url: {
                  url: `data:image/jpeg;base64,${base64Image}`,
                  detail: "high"
                }
              }
            ]
          }
        ],
        max_tokens: 500,
        temperature: 0.3, // Lower temperature for more consistent technical analysis
      });

      const analysisText = response.choices[0].message.content;
      return this.parseAIResponse(analysisText);
      
    } catch (error) {
      console.error('AI Image Analysis Error:', error);
      return this.getFallbackAnalysis();
    }
  }

  /**
   * Refine camera settings based on user's natural language input
   * @param {string} userInput - User's refinement request
   * @param {Object} currentSettings - Current camera settings
   * @returns {Promise<Object>} Refined camera settings
   */
  async refineSettings(userInput, currentSettings) {
    if (!this.isConfiguredFlag) {
      return this.getFallbackRefinement(userInput, currentSettings);
    }

    try {
      const prompt = PHOTOGRAPHY_PROMPTS.REFINEMENT_ANALYSIS
        .replace('{currentISO}', currentSettings.iso)
        .replace('{currentAperture}', currentSettings.aperture)
        .replace('{currentShutterSpeed}', currentSettings.shutterSpeed)
        .replace('{userInput}', userInput);

      const response = await openai.chat.completions.create({
        model: 'gpt-4',
        messages: [
          {
            role: "user",
            content: prompt
          }
        ],
        max_tokens: 300,
        temperature: 0.4, // Slightly higher for more creative interpretation
      });

      const refinementText = response.choices[0].message.content;
      return this.parseAIResponse(refinementText);
      
    } catch (error) {
      console.error('AI Refinement Error:', error);
      return this.getFallbackRefinement(userInput, currentSettings);
    }
  }

  /**
   * Parse AI response and validate camera settings
   * @param {string} responseText - Raw AI response
   * @returns {Object} Parsed and validated settings
   */
  parseAIResponse(responseText) {
    try {
      // Extract JSON from response (handle cases where AI adds extra text)
      const jsonMatch = responseText.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No JSON found in AI response');
      }

      const parsed = JSON.parse(jsonMatch[0]);
      
      // Validate and normalize settings
      return {
        iso: this.validateISO(parsed.iso),
        aperture: this.validateAperture(parsed.aperture),
        shutterSpeed: this.validateShutterSpeed(parsed.shutterSpeed),
        explanations: {
          iso: parsed.explanations?.iso || "ISO setting optimized for current conditions.",
          aperture: parsed.explanations?.aperture || "Aperture chosen for desired depth of field.",
          shutterSpeed: parsed.explanations?.shutterSpeed || "Shutter speed selected for proper exposure."
        },
        reasoning: parsed.reasoning || "Settings adjusted based on your creative vision.",
        tip: parsed.tip || null,
        confidence: parsed.confidence || 0.8,
        photographyType: parsed.photographyType || 'other',
        lightingCondition: parsed.lightingCondition || 'mixed'
      };
      
    } catch (error) {
      console.error('Failed to parse AI response:', error);
      return this.getFallbackAnalysis();
    }
  }

  /**
   * Validate and normalize ISO value
   */
  validateISO(iso) {
    const numISO = parseInt(iso);
    if (isNaN(numISO) || numISO < CAMERA_SETTINGS.ISO.MIN || numISO > CAMERA_SETTINGS.ISO.MAX) {
      return 100; // Default fallback
    }
    return numISO;
  }

  /**
   * Validate and normalize aperture value
   */
  validateAperture(aperture) {
    const apertureStr = aperture.toString();
    if (!apertureStr.startsWith('f/')) {
      return `f/${apertureStr}`;
    }
    return apertureStr;
  }

  /**
   * Validate and normalize shutter speed value
   */
  validateShutterSpeed(shutterSpeed) {
    const speedStr = shutterSpeed.toString();
    // Ensure proper formatting
    if (speedStr.includes('"') || speedStr.includes('s')) {
      return speedStr;
    }
    if (parseFloat(speedStr) >= 1) {
      return `${speedStr}"`;
    } else {
      return `1/${Math.round(1/parseFloat(speedStr))}s`;
    }
  }

  /**
   * Fallback analysis when AI is not configured
   */
  getFallbackAnalysis() {
    return {
      iso: 100,
      aperture: 'f/4.0',
      shutterSpeed: '1/250s',
      explanations: {
        iso: "This low setting is great for bright light and ensures your photo is sharp and noise-free.",
        aperture: "A balanced setting that keeps most of your scene in focus.",
        shutterSpeed: "Fast enough to freeze motion for a crisp shot of a person or pet."
      },
      reasoning: "These are balanced settings that work well for most general photography situations.",
      tip: null,
      confidence: 0.5,
      photographyType: 'other',
      lightingCondition: 'mixed'
    };
  }

  /**
   * Fallback refinement when AI is not configured
   */
  getFallbackRefinement(userInput, currentSettings) {
    const input = userInput.toLowerCase();
    
    // Simple keyword-based fallback with conversational responses
    if (input.includes('blur') || input.includes('bokeh')) {
      return {
        ...currentSettings,
        aperture: 'f/1.8',
        shutterSpeed: '1/1000s',
        explanations: {
          iso: "Low ISO for maximum image quality.",
          aperture: "Wide aperture creates beautiful background blur.",
          shutterSpeed: "Fast shutter speed balances the wide aperture."
        },
        reasoning: "Perfect! I've opened up the aperture to f/1.8 which will create that beautiful blurry background effect you're looking for. The faster shutter speed compensates for the extra light coming in."
      };
    }
    
    if (input.includes('sharp') || input.includes('clear')) {
      return {
        ...currentSettings,
        aperture: 'f/8.0',
        shutterSpeed: '1/125s',
        explanations: {
          iso: "Low ISO for clean, sharp results.",
          aperture: "Smaller aperture increases depth of field for sharper images.",
          shutterSpeed: "Moderate shutter speed prevents camera shake."
        },
        reasoning: "Great choice! I've closed down the aperture to f/8 which will give you much sharper images with everything in focus. This is perfect for landscapes or when you want maximum detail."
      };
    }
    
    return {
      ...currentSettings,
      reasoning: "I've made some adjustments based on your request. These settings should help you achieve the look you're going for!"
    };
  }

  /**
   * Check if AI service is properly configured
   */
  isConfigured() {
    return this.isConfiguredFlag;
  }
}

// Export singleton instance
const aiService = new AIService();
module.exports = aiService;
