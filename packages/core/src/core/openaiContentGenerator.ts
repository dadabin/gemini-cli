/**
 * @license
 * Copyright 2026 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-type-assertion */

import type {
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensResponse,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
  Part,
} from '@google/genai';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';

export class OpenAIContentGenerator implements ContentGenerator {
  constructor(
    private readonly baseUrl: string,
    private readonly modelName: string,
    private readonly apiKey: string,
    private readonly temperature?: number,
  ) {}

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const openaiRequest = this.convertToOpenAIRequest(request);

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(openaiRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`,
      );
    }

    const data = (await response.json()) as Record<string, unknown>;
    return this.convertToGeminiResponse(data);
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
    _role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const openaiRequest = this.convertToOpenAIRequest(request);
    openaiRequest['stream'] = true;

    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(openaiRequest),
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        `OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`,
      );
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    const createGeminiTextResponse = this.createGeminiTextResponse.bind(this);
    const createGeminiToolCallResponse =
      this.createGeminiToolCallResponse.bind(this);

    async function* generateStream() {
      let buffer = '';
      let currentToolCall: {
        id?: string;
        name: string;
        arguments: string;
      } | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmedLine = line.trim();
          if (
            trimmedLine.startsWith('data: ') &&
            trimmedLine !== 'data: [DONE]'
          ) {
            const dataStr = trimmedLine.slice(6);
            try {
              const data = JSON.parse(dataStr) as {
                choices?: Array<{
                  delta?: {
                    tool_calls?: Array<{
                      id?: string;
                      function?: { name?: string; arguments?: string };
                    }>;
                    content?: string;
                  };
                  finish_reason?: string;
                }>;
              };
              const choice = data.choices?.[0];
              const delta = choice?.delta;

              if (delta?.tool_calls) {
                for (const toolCall of delta.tool_calls) {
                  if (toolCall.function?.name) {
                    if (currentToolCall) {
                      yield createGeminiToolCallResponse(currentToolCall);
                    }
                    currentToolCall = {
                      id: toolCall.id,
                      name: toolCall.function.name,
                      arguments: toolCall.function.arguments || '',
                    };
                  } else if (toolCall.function?.arguments && currentToolCall) {
                    currentToolCall.arguments += toolCall.function.arguments;
                  }
                }
              } else if (delta?.content) {
                yield createGeminiTextResponse(delta.content);
              }

              if (choice?.finish_reason) {
                if (currentToolCall) {
                  yield createGeminiToolCallResponse(currentToolCall);
                  currentToolCall = null;
                }
                yield {
                  candidates: [
                    {
                      content: { role: 'model', parts: [] },
                      finishReason:
                        choice.finish_reason === 'tool_calls'
                          ? 'STOP'
                          : choice.finish_reason.toUpperCase(),
                    },
                  ],
                } as unknown as GenerateContentResponse;
              }
            } catch (_e) {
              // Ignore parse errors for incomplete chunks
            }
          }
        }
      }

      if (currentToolCall) {
        yield createGeminiToolCallResponse(currentToolCall);
      }
    }

    return generateStream();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Basic estimation: 1 token ~= 4 characters for English text
    let totalCharacters = 0;
    if (request.contents) {
      const contentsArray = Array.isArray(request.contents)
        ? request.contents
        : [request.contents];
      for (const content of contentsArray) {
        if (typeof content === 'string') {
          totalCharacters += content.length;
        } else if (
          content &&
          typeof content === 'object' &&
          'parts' in content
        ) {
          const partsArray = Array.isArray(content.parts)
            ? content.parts
            : [content.parts];
          for (const part of partsArray) {
            const p = part as unknown;
            if (typeof p === 'string') {
              totalCharacters += p.length;
            } else if (
              p &&
              typeof p === 'object' &&
              'text' in p &&
              typeof (p as { text: unknown }).text === 'string'
            ) {
              totalCharacters += (p as { text: string }).text.length;
            }
          }
        }
      }
    }
    return { totalTokens: Math.ceil(totalCharacters / 4) };
  }

  async embedContent(
    _request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    throw new Error(
      'embedContent is not supported for OpenAI compatible models yet.',
    );
  }

  private convertToOpenAIRequest(
    request: GenerateContentParameters,
  ): Record<string, unknown> {
    const messages: Array<Record<string, unknown>> = [];

    if (request.config?.systemInstruction) {
      let systemContent = '';
      if (typeof request.config.systemInstruction === 'string') {
        systemContent = request.config.systemInstruction;
      } else if (
        typeof request.config.systemInstruction === 'object' &&
        request.config.systemInstruction !== null &&
        'parts' in request.config.systemInstruction
      ) {
        const parts = (request.config.systemInstruction as { parts: Part[] })
          .parts;
        systemContent = parts.map((p: Part) => p.text || '').join('');
      }
      messages.push({ role: 'system', content: systemContent });
    }

    if (request.contents) {
      const contentsArray = Array.isArray(request.contents)
        ? request.contents
        : [request.contents];
      for (const content of contentsArray) {
        if (typeof content === 'string') {
          messages.push({
            role: 'user',
            content,
          });
        } else if (
          content &&
          typeof content === 'object' &&
          'parts' in content
        ) {
          const role =
            'role' in content && (content as { role?: string }).role === 'model'
              ? 'assistant'
              : 'user';
          const partsArray = Array.isArray(content.parts)
            ? content.parts
            : [content.parts];

          const toolCallParts = partsArray.filter(
            (p: unknown) => p && typeof p === 'object' && 'functionCall' in p,
          );
          const toolResponseParts = partsArray.filter(
            (p: unknown) =>
              p && typeof p === 'object' && 'functionResponse' in p,
          );
          const textParts = partsArray.filter(
            (p: unknown) => p && typeof p === 'object' && 'text' in p,
          );

          if (toolCallParts.length > 0) {
            messages.push({
              role: 'assistant',
              content:
                textParts
                  .map((p: unknown) => (p as { text?: string }).text)
                  .join('') || null,
              tool_calls: toolCallParts.map((p: unknown) => {
                const fnCall = (
                  p as {
                    functionCall?: {
                      id?: string;
                      name?: string;
                      args?: unknown;
                    };
                  }
                ).functionCall;
                return {
                  id: fnCall?.id || fnCall?.name || 'call_id',
                  type: 'function',
                  function: {
                    name: fnCall?.name,
                    arguments: JSON.stringify(fnCall?.args || {}),
                  },
                };
              }),
            });
          } else if (toolResponseParts.length > 0) {
            for (const p of toolResponseParts) {
              const fnResp = (
                p as {
                  functionResponse?: {
                    id?: string;
                    name?: string;
                    response?: unknown;
                  };
                }
              ).functionResponse;
              if (!fnResp) continue;
              messages.push({
                role: 'tool',
                tool_call_id: fnResp.id || fnResp.name || 'call_id',
                name: fnResp.name,
                content: JSON.stringify(fnResp.response || {}),
              });
            }
          } else {
            const contentParts: Array<Record<string, unknown>> = [];
            for (const p of partsArray) {
              if (typeof p === 'string') {
                contentParts.push({ type: 'text', text: p });
              } else if (p && typeof p === 'object') {
                if ('text' in p) {
                  contentParts.push({
                    type: 'text',
                    text: (p as { text?: string }).text,
                  });
                } else if ('inlineData' in p) {
                  const inlineData = (
                    p as { inlineData?: { mimeType?: string; data?: string } }
                  ).inlineData;
                  if (inlineData) {
                    contentParts.push({
                      type: 'image_url',
                      image_url: {
                        url: `data:${inlineData.mimeType};base64,${inlineData.data}`,
                      },
                    });
                  }
                }
              }
            }
            if (
              contentParts.length === 1 &&
              contentParts[0]?.['type'] === 'text'
            ) {
              messages.push({
                role,
                content: contentParts[0]?.['text'],
              });
            } else if (contentParts.length > 0) {
              messages.push({
                role,
                content: contentParts,
              });
            }
          }
        }
      }
    }

    const openaiRequest: Record<string, unknown> = {
      model: request.model || this.modelName,
      messages,
    };

    if (this.temperature !== undefined) {
      openaiRequest['temperature'] = this.temperature;
    }

    // Map generation config
    if (request.config) {
      if (request.config.maxOutputTokens !== undefined) {
        openaiRequest['max_tokens'] = request.config.maxOutputTokens;
      }
      if (request.config.temperature !== undefined) {
        openaiRequest['temperature'] = request.config.temperature;
      }
      // if (request.config.topP !== undefined) {
      //   openaiRequest['top_p'] = request.config.topP;
      // }
      if (request.config.stopSequences !== undefined) {
        openaiRequest['stop'] = request.config.stopSequences;
      }
      if (request.config.frequencyPenalty !== undefined) {
        openaiRequest['frequency_penalty'] = request.config.frequencyPenalty;
      }
      if (request.config.presencePenalty !== undefined) {
        openaiRequest['presence_penalty'] = request.config.presencePenalty;
      }
    }

    if (request.config?.tools) {
      const openaiTools: Array<Record<string, unknown>> = [];
      for (const tool of request.config.tools) {
        if ('functionDeclarations' in tool && tool.functionDeclarations) {
          const functionDeclarations = tool.functionDeclarations as Array<{
            name: string;
            description?: string;
            parameters?: unknown;
          }>;
          for (const func of functionDeclarations) {
            openaiTools.push({
              type: 'function',
              function: {
                name: func.name,
                description: func.description,
                parameters: func.parameters,
              },
            });
          }
        }
      }
      if (openaiTools.length > 0) {
        openaiRequest['tools'] = openaiTools;
      }
    }

    return openaiRequest;
  }

  private convertToGeminiResponse(
    data: Record<string, unknown>,
  ): GenerateContentResponse {
    const choice = (data['choices'] as Array<Record<string, unknown>>)?.[0];
    const message = choice?.['message'] as Record<string, unknown> | undefined;

    const parts: Part[] = [];
    if (message?.['content']) {
      parts.push({ text: message['content'] as string });
    }

    if (message?.['tool_calls']) {
      const toolCalls = message['tool_calls'] as Array<Record<string, unknown>>;
      for (const toolCall of toolCalls) {
        if (toolCall['type'] === 'function') {
          const functionData = toolCall['function'] as Record<string, unknown>;
          parts.push({
            functionCall: {
              id: toolCall['id'] as string | undefined,
              name: functionData['name'] as string,
              args: JSON.parse((functionData['arguments'] as string) || '{}'),
            },
          });
        }
      }
    }

    const finishReason = choice?.['finish_reason'] as string | undefined;

    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason:
            finishReason === 'tool_calls'
              ? 'STOP'
              : finishReason?.toUpperCase() || 'STOP',
        },
      ],
      usageMetadata: {
        promptTokenCount:
          ((data['usage'] as Record<string, unknown>)?.[
            'prompt_tokens'
          ] as number) || 0,
        candidatesTokenCount:
          ((data['usage'] as Record<string, unknown>)?.[
            'completion_tokens'
          ] as number) || 0,
        totalTokenCount:
          ((data['usage'] as Record<string, unknown>)?.[
            'total_tokens'
          ] as number) || 0,
      },
    } as unknown as GenerateContentResponse;
  }

  private createGeminiTextResponse(text: string): GenerateContentResponse {
    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [{ text }],
          },
        },
      ],
    } as unknown as GenerateContentResponse;
  }

  private createGeminiToolCallResponse(toolCall: {
    id?: string;
    name: string;
    arguments: string;
  }): GenerateContentResponse {
    let args = {};
    try {
      args = JSON.parse(toolCall.arguments || '{}');
    } catch (_e) {
      // If parsing fails, it might be incomplete, but we try our best
    }
    return {
      candidates: [
        {
          content: {
            role: 'model',
            parts: [
              {
                functionCall: {
                  id: toolCall.id,
                  name: toolCall.name,
                  args,
                },
              },
            ],
          },
        },
      ],
    } as unknown as GenerateContentResponse;
  }
}
