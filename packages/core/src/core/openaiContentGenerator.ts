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
    const mapOpenAIFinishReason = this.mapOpenAIFinishReason.bind(this);

    async function* generateStream() {
      let buffer = '';
      const toolCallsByIndex = new Map<
        number,
        {
          id?: string;
          name: string;
          arguments: string;
        }
      >();

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
                      index?: number;
                      id?: string;
                      function?: { name?: string; arguments?: string };
                    }>;
                    function_call?: { name?: string; arguments?: string };
                    content?: string;
                  };
                  finish_reason?: string;
                }>;
              };
              const choice = data.choices?.[0];
              const delta = choice?.delta;

              if (delta?.tool_calls) {
                for (const toolCall of delta.tool_calls) {
                  const index = toolCall.index ?? 0;
                  if (toolCall.function?.name) {
                    toolCallsByIndex.set(index, {
                      id: toolCall.id,
                      name: toolCall.function.name,
                      arguments: toolCall.function.arguments || '',
                    });
                  } else if (
                    toolCall.function?.arguments &&
                    toolCallsByIndex.has(index)
                  ) {
                    const current = toolCallsByIndex.get(index)!;
                    current.arguments += toolCall.function.arguments;
                  }
                }
              } else if (delta?.function_call) {
                const index = 0; // function_call doesn't have index, usually
                if (delta.function_call.name) {
                  toolCallsByIndex.set(index, {
                    name: delta.function_call.name,
                    arguments: delta.function_call.arguments || '',
                  });
                } else if (
                  delta.function_call.arguments &&
                  toolCallsByIndex.has(index)
                ) {
                  const current = toolCallsByIndex.get(index)!;
                  current.arguments += delta.function_call.arguments;
                }
              } else if (
                delta?.content !== undefined &&
                delta?.content !== null
              ) {
                yield createGeminiTextResponse(delta.content);
              }

              if (choice?.finish_reason) {
                if (toolCallsByIndex.size > 0) {
                  // Sort by index to ensure deterministic order
                  const sortedIndices = Array.from(
                    toolCallsByIndex.keys(),
                  ).sort((a, b) => a - b);
                  for (const index of sortedIndices) {
                    yield createGeminiToolCallResponse(
                      toolCallsByIndex.get(index)!,
                    );
                  }
                  toolCallsByIndex.clear();
                }

                // Don't yield a STOP reason if we had tool calls, let the tool calls dictate the next turn
                const mappedReason = mapOpenAIFinishReason(
                  choice.finish_reason,
                );
                yield {
                  candidates: [
                    {
                      content: { role: 'model', parts: [] },
                      finishReason: mappedReason,
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

      if (toolCallsByIndex.size > 0) {
        const sortedIndices = Array.from(toolCallsByIndex.keys()).sort(
          (a, b) => a - b,
        );
        for (const index of sortedIndices) {
          yield createGeminiToolCallResponse(toolCallsByIndex.get(index)!);
        }
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

    let pendingToolCallIds: Record<string, string[]> = {};

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
            pendingToolCallIds = {}; // Reset for new assistant message
            const toolCalls = toolCallParts.map((p: unknown, index: number) => {
              const fnCall = (
                p as {
                  functionCall?: {
                    id?: string;
                    name?: string;
                    args?: unknown;
                  };
                }
              ).functionCall;

              const generatedId =
                fnCall?.id || `call_${fnCall?.name || 'tool'}_${index}`;
              if (fnCall?.name) {
                if (!pendingToolCallIds[fnCall.name]) {
                  pendingToolCallIds[fnCall.name] = [];
                }
                pendingToolCallIds[fnCall.name].push(generatedId);
              }

              return {
                id: generatedId,
                type: 'function',
                function: {
                  name: fnCall?.name,
                  arguments: JSON.stringify(fnCall?.args || {}),
                },
              };
            });

            const contentText = textParts
              .map((p: unknown) => (p as { text?: string }).text)
              .join('');

            // If using Anthropic/Claude through an OpenAI proxy, empty text with tool calls might cause issues.
            // Some proxies require content to be explicitly null or empty string, or omit it entirely.
            // OpenAI officially allows null or empty string when tool_calls is present.
            // Using empty string is generally more compatible with various proxies.
            messages.push({
              role: 'assistant',
              content: contentText || '',
              tool_calls: toolCalls,
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

              let toolCallId = fnResp.id;
              if (
                fnResp.name &&
                pendingToolCallIds[fnResp.name] &&
                pendingToolCallIds[fnResp.name].length > 0
              ) {
                toolCallId = pendingToolCallIds[fnResp.name].shift();
              } else if (!toolCallId) {
                toolCallId = `call_${fnResp.name}_0`;
              }

              // The content MUST be a string. If the tool response is empty, it should be "{}"
              let contentString = '{}';
              if (fnResp.response) {
                contentString =
                  typeof fnResp.response === 'string'
                    ? fnResp.response
                    : JSON.stringify(fnResp.response);
              }

              messages.push({
                role: 'tool',
                tool_call_id: toolCallId,
                content: contentString,
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

    // Second pass: ensure all tool_calls have corresponding tool messages
    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      if (msg['role'] === 'assistant' && msg['tool_calls']) {
        const requiredToolCallIds = new Set(
          (msg['tool_calls'] as Array<{ id: string }>).map((tc) => tc.id),
        );

        let j = i + 1;
        while (j < messages.length && messages[j]['role'] === 'tool') {
          requiredToolCallIds.delete(messages[j]['tool_call_id'] as string);
          j++;
        }

        if (requiredToolCallIds.size > 0) {
          for (const missingId of requiredToolCallIds) {
            messages.splice(j, 0, {
              role: 'tool',
              tool_call_id: missingId,
              content: JSON.stringify({
                error:
                  'Tool execution was interrupted or failed to return output.',
              }),
            });
            j++;
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
      if (request.config.topP !== undefined) {
        openaiRequest['top_p'] = request.config.topP;
      }
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
            // OpenAI requires parameters to be defined, even if empty
            // "parameters": { "type": "object", "properties": {} }
            let parameters = func.parameters;
            if (!parameters) {
              parameters = {
                type: 'object',
                properties: {},
              };
            }

            openaiTools.push({
              type: 'function',
              function: {
                name: func.name,
                description: func.description,
                parameters,
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
              id: toolCall['id'] as string,
              name: functionData['name'] as string,
              args: JSON.parse((functionData['arguments'] as string) || '{}'),
            },
          });
        }
      }
    } else if (message?.['function_call']) {
      const functionData = message['function_call'] as Record<string, unknown>;
      parts.push({
        functionCall: {
          name: functionData['name'] as string,
          args: JSON.parse((functionData['arguments'] as string) || '{}'),
        },
      });
    }

    const finishReason = choice?.['finish_reason'] as string | undefined;
    const functionCalls = parts
      .filter((p) => p.functionCall)
      .map((p) => p.functionCall!);

    return {
      functionCalls: functionCalls.length > 0 ? functionCalls : undefined,
      candidates: [
        {
          content: {
            role: 'model',
            parts,
          },
          finishReason: this.mapOpenAIFinishReason(finishReason),
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

  private mapOpenAIFinishReason(finishReason?: string): string {
    if (!finishReason) return 'STOP';
    switch (finishReason.toLowerCase()) {
      case 'stop':
        return 'STOP';
      case 'length':
      case 'max_tokens':
        return 'MAX_TOKENS';
      case 'content_filter':
        return 'SAFETY';
      case 'tool_calls':
      case 'function_call':
        return 'STOP';
      default:
        return finishReason.toUpperCase();
    }
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
    const functionCall = {
      id: toolCall.id,
      name: toolCall.name,
      args,
    };
    return {
      functionCalls: [functionCall],
      candidates: [
        {
          content: {
            role: 'model',
            parts: [
              {
                functionCall,
              },
            ],
          },
        },
      ],
    } as unknown as GenerateContentResponse;
  }
}
