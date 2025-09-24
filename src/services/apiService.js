// API Service for communicating with backend
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:3001/api';

class APIService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  /**
   * Check if backend is healthy and AI is configured
   * @returns {Promise<Object>} Health status
   */
  async checkHealth() {
    try {
      const response = await fetch(`${this.baseURL}/health`);
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Health check failed:', error);
      return {
        status: 'unhealthy',
        aiConfigured: false,
        error: error.message
      };
    }
  }

  /**
   * Analyze uploaded image and get DSLR settings
   * @param {File} imageFile - The uploaded image file
   * @returns {Promise<Object>} Camera settings and analysis
   */
  async analyzeImage(imageFile) {
    try {
      const formData = new FormData();
      formData.append('image', imageFile);

      const response = await fetch(`${this.baseURL}/analyze-image`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Analysis failed: ${response.status}`);
      }

      const result = await response.json();
      return result.data;
      
    } catch (error) {
      console.error('Image analysis failed:', error);
      throw error;
    }
  }

  /**
   * Refine camera settings based on user input
   * @param {string} userInput - User's refinement request
   * @param {Object} currentSettings - Current camera settings
   * @returns {Promise<Object>} Refined camera settings
   */
  async refineSettings(userInput, currentSettings) {
    try {
      const response = await fetch(`${this.baseURL}/refine-settings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userInput,
          currentSettings
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Refinement failed: ${response.status}`);
      }

      const result = await response.json();
      return result.data;
      
    } catch (error) {
      console.error('Settings refinement failed:', error);
      throw error;
    }
  }
}

// Export singleton instance
const apiService = new APIService();
export default apiService;
