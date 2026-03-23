import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import { api } from './api';
import './App.css';

function App() {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
  }, []);

  // Load conversation details when selected
  useEffect(() => {
    if (currentConversationId) {
      loadConversation(currentConversationId);
    }
  }, [currentConversationId]);

  const loadConversations = async () => {
    try {
      const convs = await api.listConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadConversation = async (id) => {
    try {
      const conv = await api.getConversation(id);
      setCurrentConversation(conv);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const handleNewConversation = async () => {
    try {
      const newConv = await api.createConversation();
      setConversations([
        { id: newConv.id, created_at: newConv.created_at, message_count: 0 },
        ...conversations,
      ]);
      setCurrentConversationId(newConv.id);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = (id) => {
    setCurrentConversationId(id);
  };

  /**
   * Helper to immutably update the last assistant message in the conversation.
   * Avoids direct state mutation by creating new object references.
   */
  const updateLastAssistantMessage = (updater) => {
    setCurrentConversation((prev) => {
      const messages = prev.messages.map((msg, i) => {
        if (i === prev.messages.length - 1) {
          // Deep clone the last message to avoid mutation
          const cloned = {
            ...msg,
            loading: { ...msg.loading },
            metadata: msg.metadata ? { ...msg.metadata } : null,
          };
          updater(cloned);
          return cloned;
        }
        return msg;
      });
      return { ...prev, messages };
    });
  };

  const handleSendMessage = async (content) => {
    if (!currentConversationId) return;

    setIsLoading(true);
    try {
      // Optimistically add user message to UI
      const userMessage = { role: 'user', content };
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

      // Create a partial assistant message that will be updated progressively
      const assistantMessage = {
        role: 'assistant',
        stage1: null,
        stage2: null,
        stage3: null,
        metadata: null,
        loading: {
          stage1: false,
          stage2: false,
          stage3: false,
        },
      };

      // Add the partial assistant message
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, assistantMessage],
      }));

      // Send message with streaming
      await api.sendMessageStream(currentConversationId, content, (eventType, event) => {
        switch (eventType) {
          case 'stage1_start':
            updateLastAssistantMessage((msg) => {
              msg.loading.stage1 = true;
            });
            break;

          case 'stage1_complete':
            updateLastAssistantMessage((msg) => {
              msg.stage1 = event.data;
              msg.loading.stage1 = false;
            });
            break;

          case 'stage2_start':
            updateLastAssistantMessage((msg) => {
              msg.loading.stage2 = true;
            });
            break;

          case 'stage2_complete':
            updateLastAssistantMessage((msg) => {
              msg.stage2 = event.data;
              msg.metadata = event.metadata;
              msg.loading.stage2 = false;
            });
            break;

          case 'stage3_start':
            updateLastAssistantMessage((msg) => {
              msg.loading.stage3 = true;
            });
            break;

          case 'stage3_complete':
            updateLastAssistantMessage((msg) => {
              msg.stage3 = event.data;
              msg.loading.stage3 = false;
            });
            break;

          case 'title_complete':
            loadConversations();
            break;

          case 'complete':
            loadConversations();
            setIsLoading(false);
            break;

          case 'error':
            console.error('Stream error:', event.message);
            setIsLoading(false);
            break;

          default:
            console.log('Unknown event type:', eventType);
        }
      });
    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove optimistic messages on error
      setCurrentConversation((prev) => ({
        ...prev,
        messages: prev.messages.slice(0, -2),
      }));
      setIsLoading(false);
    }
  };

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
      />
      <ChatInterface
        conversation={currentConversation}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
      />
    </div>
  );
}

export default App;
