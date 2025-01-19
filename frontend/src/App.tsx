import { useState } from 'react'

interface Card {
  id: string;
  text: string;
  audioUrl: string;
}

function App() {
  const [cards, setCards] = useState<Card[]>([]);
  const [inputText, setInputText] = useState('');

  const handleSubmit = async () => {
    if (!inputText.trim()) return;

    try {
      const response = await fetch('http://localhost:8000/api/text-to-speech', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: inputText,
          voice: 'alloy'
        }),
      });

      if (!response.ok) throw new Error('Failed to generate audio');

      const data = await response.json();
      
      const newCard: Card = {
        id: crypto.randomUUID(),
        text: inputText,
        audioUrl: `data:audio/wav;base64,${data.audio_data}`
      };

      setCards(prev => [...prev, newCard]);
      setInputText('');
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to generate audio');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900">
      <div className="container mx-auto p-4">
        <div className="grid grid-cols-3 gap-4 h-[calc(100vh-2rem)]">
          {/* Left Panel - Input */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <div className="space-y-4">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="w-full h-48 p-2 border rounded-md dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                placeholder="Enter text here..."
              />
              <button
                onClick={handleSubmit}
                className="w-full px-4 py-2 text-white bg-blue-500 rounded-md hover:bg-blue-600 transition-colors"
                disabled={!inputText.trim()}
              >
                Generate Card
              </button>
            </div>
          </div>

          {/* Middle Panel - Cards */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 overflow-y-auto">
            <div className="space-y-4">
              {cards.map(card => (
                <div key={card.id} className="p-4 border rounded-lg dark:border-gray-700">
                  <p className="text-gray-800 dark:text-gray-200 mb-2">{card.text}</p>
                  <audio controls className="w-full">
                    <source src={card.audioUrl} type="audio/wav" />
                    Your browser does not support the audio element.
                  </audio>
                </div>
              ))}
            </div>
          </div>

          {/* Right Panel - Reserved for future use */}
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
            <p className="text-gray-500 dark:text-gray-400 text-center">
              Future Features
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
