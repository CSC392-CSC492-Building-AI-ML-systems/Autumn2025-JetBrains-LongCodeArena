"""
Different prompt templates for module summarization task.
Test different prompts to improve the quality of generation.
"""

def get_prompt_v1_original(intent, filename, code_context):
    """
    Original baseline prompt - simple and direct
    """
    prompt = 'I have code collected from one or more files joined into one string. '
    prompt += f'Using the code generate text for {filename} file with documentation about {intent}.\n\n'
    prompt += f'My code:\n\n{code_context}'
    prompt += f'\n\n\n\nAs answer return text for {filename} file about {intent}. Do not return the instruction how to make documentation, return only documentation itself.'
    return prompt


def get_prompt_v2_structured(intent, filename, code_context):
    """
    Structured prompt with clear sections and better formatting
    """
    prompt = f"""You are an expert technical writer. Your task is to generate high-quality documentation.

**Task**: Generate documentation for the file `{filename}` that describes: {intent}

**Source Code**:
```
{code_context}
```

**Requirements**:
1. Write clear, concise documentation focusing on {intent}
2. Use proper markdown formatting
3. Include relevant code examples if applicable
4. Explain the purpose, usage, and key functionality
5. Do NOT include meta-instructions or explanations about how you wrote the documentation

**Output**: Return only the documentation content for `{filename}`.
"""
    return prompt


def get_prompt_v3_detailed(intent, filename, code_context):
    """
    More detailed prompt with specific documentation guidelines
    """
    prompt = f"""You are a senior software engineer writing technical documentation.

# Documentation Generation Task

## Target File: `{filename}`
## Topic: {intent}

## Source Code:
{code_context}

## Instructions:
Generate comprehensive documentation for `{filename}` that covers {intent}. Your documentation should:

### Content Requirements:
- Provide a clear overview of the module's purpose
- Explain key functions, classes, and their relationships
- Describe important parameters and return values
- Include usage examples where appropriate
- Highlight any important considerations or edge cases

### Style Guidelines:
- Use clear, professional language
- Follow markdown formatting conventions
- Be concise but thorough
- Focus on practical information for developers
- Avoid redundant explanations

### Output Format:
- Start with a brief introduction
- Use appropriate headers for organization
- Include code snippets in markdown code blocks
- End with usage examples if relevant

**Important**: Return ONLY the documentation content itself. Do not include any meta-commentary, instructions, or explanations about the documentation process.
"""
    return prompt


def get_prompt_v4_role_based(intent, filename, code_context):
    """
    Role-based prompt emphasizing understanding and explaining
    """
    prompt = f"""Act as an experienced software documentation specialist who deeply understands code architecture and can explain it clearly to other developers.

Your mission: Analyze the provided code and create professional documentation for `{filename}` focusing on: {intent}

## Code to Document:
{code_context}

## Your Approach:
1. First, identify the core purpose and functionality in the code
2. Determine what developers need to know about {intent}
3. Structure the information logically and clearly
4. Write documentation that is both accurate and easy to understand

## Documentation Standards:
- Be precise and technically accurate
- Use industry-standard markdown formatting
- Include practical examples derived from the code
- Explain complex concepts clearly
- Focus on what matters for {intent}

Generate the documentation for `{filename}` now. Output only the final documentation text.
"""
    return prompt


def get_prompt_v5_example_guided(intent, filename, code_context):
    """
    Prompt with example structure to guide the output format
    """
    prompt = f"""Generate technical documentation for the file `{filename}` about {intent}.

## Source Code:
{code_context}

## Expected Documentation Structure:

# [Title based on {intent}]

## Overview
[Brief description of what this module does]

## Key Components
[Description of main functions/classes/features]

## Usage
[How to use this module, with examples if applicable]

## Details
[Important information about {intent}]

---

**Instructions**:
- Follow the structure above
- Write clear, professional documentation
- Use markdown formatting
- Focus specifically on {intent}
- Include code examples from the provided source when helpful
- Be concise and informative

**Output only the documentation content for `{filename}` - no meta-commentary.**
"""
    return prompt


def get_prompt_v6_concise(intent, filename, code_context):
    """
    Shorter, more direct prompt for testing if brevity improves output
    """
    prompt = f"""Create technical documentation for `{filename}` about {intent}.

Code:
{code_context}

Write clear, well-formatted markdown documentation that:
- Explains {intent} thoroughly
- Includes relevant examples
- Uses proper structure and formatting
- Is aimed at developers using this code

Return only the documentation content.
"""
    return prompt


def get_prompt_v7_quality_focused(intent, filename, code_context):
    """
    Prompt emphasizing documentation quality and best practices
    """
    prompt = f"""You are creating high-quality technical documentation that will be used by professional developers.

**Target**: `{filename}`
**Focus**: {intent}

**Source Code**:
```
{code_context}
```

**Quality Standards**:
✓ Accuracy: All information must be correct and derived from the code
✓ Clarity: Use simple, direct language without jargon when possible
✓ Completeness: Cover all important aspects of {intent}
✓ Usefulness: Include practical information and examples
✓ Organization: Use clear structure with appropriate headers
✓ Formatting: Apply proper markdown conventions

**Style**:
- Write in present tense
- Use active voice
- Be specific and concrete
- Include code examples when they add value
- Focus on the "what" and "how", not the "why you should document"

Generate professional documentation for `{filename}`. Return only the documentation text itself.
"""
    return prompt


# Prompt registry for easy access
PROMPT_VERSIONS = {
    'v1_original': get_prompt_v1_original,
    'v2_structured': get_prompt_v2_structured,
    'v3_detailed': get_prompt_v3_detailed,
    'v4_role_based': get_prompt_v4_role_based,
    'v5_example_guided': get_prompt_v5_example_guided,
    'v6_concise': get_prompt_v6_concise,
    'v7_quality_focused': get_prompt_v7_quality_focused,
}


def get_prompt(version, intent, filename, code_context):
    """
    Get a prompt by version name.
    
    Args:
        version: One of 'v1_original', 'v2_structured', 'v3_detailed', 
                'v4_role_based', 'v5_example_guided', 'v6_concise', 'v7_quality_focused'
        intent: Documentation intent/topic
        filename: Target documentation filename
        code_context: Source code to document
    
    Returns:
        Formatted prompt string
    """
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown prompt version: {version}. Available: {list(PROMPT_VERSIONS.keys())}")
    
    return PROMPT_VERSIONS[version](intent, filename, code_context)
