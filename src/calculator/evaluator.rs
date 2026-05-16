//! Expression evaluation module

use anyhow::{bail, Result};
use meval::{Context, Expr};
use std::collections::HashMap;
use std::sync::Arc;

use super::math_functions::*;
use super::mods::{Mod, ModManager};
use super::random::*;

/// Evaluates mathematical expressions
pub struct Evaluator {
    /// Whether to use safe evaluation mode
    safe_mode: bool,

    /// Context with custom functions
    context: Context<'static>,

    /// Mod manager for custom mod functions
    mod_manager: ModManager,

    /// Expression cache for performance optimization
    expr_cache: HashMap<String, Arc<Expr>>,

    /// Result cache for performance optimization
    result_cache: HashMap<String, f64>,

    /// Last calculation result (for 'm' constant)
    last_result: f64,
}

impl Evaluator {
    /// Creates a new evaluator
    pub fn new() -> Self {
        let mut ctx = Context::new();

        // Add mathematical constants
        ctx.var("pi", pi());
        ctx.var("e", e());
        ctx.var("m", 0.0); // Last result (memory)

        // Add trigonometric functions
        ctx.func("sin", sin);
        ctx.func("cos", cos);
        ctx.func("csin", csin);
        ctx.func("tan", tan);
        ctx.func("asin", asin);
        ctx.func("acos", acos);
        ctx.func("atan", atan);

        // Add hyperbolic functions
        ctx.func("sinh", sinh);
        ctx.func("cosh", cosh);
        ctx.func("tanh", tanh);

        // Add exponential and logarithmic functions
        ctx.func("exp", exp);
        ctx.func("sqrt", sqrt);
        ctx.funcn("pow", |args| pow(args[0], args[1]), 2);
        ctx.func("log", log);
        ctx.func("log10", log10);
        ctx.func("log2", log2);

        // Add rounding functions
        ctx.func("ceil", ceil);
        ctx.func("floor", floor);
        ctx.func("trunc", trunc);

        // Add absolute value
        ctx.func("fabs", fabs);

        // Add factorial and gamma functions
        ctx.func("factorial", factorial);
        ctx.func("gamma", gamma);

        // Add error functions
        ctx.func("erf", erf);
        ctx.func("erfc", erfc);

        // Add angle conversion functions
        ctx.func("degrees", degrees);
        ctx.func("radians", radians);

        // Add geometric functions (single argument versions)
        ctx.func("s_circle", circle_area);

        // Add geometric functions (two argument versions)
        ctx.funcn("s_tri", |args| triangle_area(args[0], args[1]), 2);
        ctx.funcn("s_rect", |args| rectangle_area(args[0], args[1]), 2);

        // Add geometric functions (three argument versions)
        ctx.funcn("s_tra", |args| trapezoid_area(args[0], args[1], args[2]), 3);
        ctx.funcn("hsf_s_tri", |args| {
            heron_triangle_area(args[0], args[1], args[2]).unwrap_or(f64::NAN)
        }, 3);
        ctx.funcn("pt", |args| pythagorean_theorem(args[0], args[1]), 2);

        // Add atan2 function
        ctx.funcn("atan2", |args| atan2(args[0], args[1]), 2);

        // Add float info functions
        ctx.func("isinf", |x| if is_inf(x) { 1.0 } else { 0.0 });
        ctx.func("isnan", |x| if is_nan(x) { 1.0 } else { 0.0 });
        ctx.funcn("isclose", |args| {
            if is_close(args[0], args[1], 1e-9, 0.0) { 1.0 } else { 0.0 }
        }, 2);
        ctx.funcn("isclose_tol", |args| {
            if is_close(args[0], args[1], args[2], args[3]) { 1.0 } else { 0.0 }
        }, 4);

        // Add GCD and LCM functions
        ctx.funcn("gcd", |args| gcd(args[0] as u64, args[1] as u64) as f64, 2);
        ctx.funcn("lcm", |args| lcm(args[0] as u64, args[1] as u64) as f64, 2);

        // Add modf function (separate integer and fractional parts)
        ctx.funcn("modf", |args| args[0].fract(), 1);

        // Add random functions
        ctx.func("random", |_| random()); // Takes dummy parameter
        ctx.funcn("randint", |args| randint(args[0] as i64, args[1] as i64) as f64, 2);
        ctx.funcn("uniform", |args| uniform(args[0], args[1]), 2);

        // Add comparison operator functions
        ctx.funcn("_cmp_eq", |args| if (args[0] - args[1]).abs() < 1e-10 { 1.0 } else { 0.0 }, 2);
        ctx.funcn("_cmp_gt", |args| if args[0] > args[1] { 1.0 } else { 0.0 }, 2);
        ctx.funcn("_cmp_lt", |args| if args[0] < args[1] { 1.0 } else { 0.0 }, 2);
        ctx.funcn("_cmp_gte", |args| if args[0] >= args[1] { 1.0 } else { 0.0 }, 2);
        ctx.funcn("_cmp_lte", |args| if args[0] <= args[1] { 1.0 } else { 0.0 }, 2);

        // Add logical operator functions
        ctx.funcn("lor", |args| if args[0] != 0.0 || args[1] != 0.0 { 1.0 } else { 0.0 }, 2);
        ctx.funcn("land", |args| if args[0] != 0.0 && args[1] != 0.0 { 1.0 } else { 0.0 }, 2);
        ctx.func("lnot", |x| if x == 0.0 { 1.0 } else { 0.0 });

        let mut mod_manager = ModManager::new();
        let _ = mod_manager.load_mods(); // Silently ignore errors if mods dir doesn't exist

        Self {
            safe_mode: true,
            context: ctx,
            mod_manager,
            expr_cache: HashMap::new(),
            result_cache: HashMap::new(),
            last_result: 0.0,
        }
    }

    /// Sets the evaluation mode
    pub fn set_safe_mode(&mut self, safe: bool) {
        self.safe_mode = safe;
    }

    /// Sets the last result (updates 'm' constant)
    pub fn set_last_result(&mut self, result: f64) {
        self.last_result = result;
        self.context.var("m", result);
    }

    /// Gets the last result
    pub fn get_last_result(&self) -> f64 {
        self.last_result
    }

    /// Reload all mods
    pub fn reload_mods(&mut self) -> Result<(), anyhow::Error> {
        self.mod_manager.reload_mods()
    }

    /// List all available mods
    pub fn list_mods(&self) -> Vec<String> {
        self.mod_manager.list_mods()
    }

    /// Get required variables for a mod
    pub fn get_required_vars(&self, name: &str) -> Option<Vec<String>> {
        self.mod_manager.get_required_vars(name)
    }

    /// Get warnings from mod loading
    pub fn get_warnings(&self) -> &[String] {
        self.mod_manager.get_warnings()
    }

    /// Clear warnings
    pub fn clear_warnings(&mut self) {
        self.mod_manager.clear_warnings();
    }

    /// Get a mod by name
    pub fn get_mod(&self, name: &str) -> Option<&Mod> {
        self.mod_manager.get_mod(name)
    }

    /// Preprocess expression to support additional operators
    fn preprocess_expression(expr: &str) -> String {
        let mut result = expr.to_string();

        // Convert ** to ^ (power operator)
        result = result.replace("**", "^");

        // Convert // to floor division: a // b -> floor(a / b)
        result = Self::replace_floor_division(&result);

        // Convert comparison operators to function calls using markers
        result = result.replace(" == ", " __CMP__eq__ ");
        result = result.replace(" >= ", " __CMP__gte__ ");
        result = result.replace(" <= ", " __CMP__lte__ ");
        result = result.replace(" > ", " __CMP__gt__ ");
        result = result.replace(" < ", " __CMP__lt__ ");

        // Process comparison markers
        result = Self::process_cmp_markers(&result);

        // Convert logical operators to function calls
        result = result.replace(" or ", " __LOG__or__ ");
        result = result.replace(" and ", " __LOG__and__ ");
        result = result.replace("not ", "__LOG__not__");

        // Process logical markers
        result = Self::process_log_markers(&result);

        result
    }

    /// Process comparison markers: "a __CMP__op__ b" -> "_op(a, b)"
    fn process_cmp_markers(expr: &str) -> String {
        let mut result = expr.to_string();

        while let Some(pos) = result.find("__CMP__") {
            // Find the operator
            let op_start = pos + 7; // length of "__CMP__"
            let op_end = result[op_start..].find("__ ").map(|p| op_start + p).unwrap_or(result.len());
            let op = &result[op_start..op_end];

            // Find the left operand (go backwards from pos)
            let left_end = pos.saturating_sub(1);
            let mut left_start = 0;
            for (idx, ch) in result[..left_end].char_indices().rev() {
                if ch == ' ' || ch == '(' || ch == ')' || ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^' {
                    left_start = idx + 1;
                    break;
                }
                left_start = idx;
            }
            let left = result[left_start..left_end].trim();

            // Find the right operand (go forwards from op_end + 3)
            let right_start = op_end + 3; // skip "__ "
            // Skip leading spaces
            let mut actual_start = right_start;
            while actual_start < result.len() && result.chars().nth(actual_start) == Some(' ') {
                actual_start += 1;
            }
            let mut right_end = result.len();
            for (idx, ch) in result[actual_start..].char_indices() {
                if ch == ' ' || ch == '(' || ch == ')' || ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^' || ch == ',' {
                    right_end = actual_start + idx;
                    break;
                }
            }
            let right = result[actual_start..right_end].trim();

            // Replace the entire pattern with function call
            let pattern_start = left_start;
            let pattern_end = right_end;
            let replacement = format!("_cmp_{}({}, {})", op, left, right);
            result.replace_range(pattern_start..pattern_end, &replacement);
        }

        result
    }

    /// Process logical markers
    fn process_log_markers(expr: &str) -> String {
        let mut result = expr.to_string();

        // Handle "not" first (unary)
        while let Some(pos) = result.find("__LOG__not__") {
            let start = pos;
            let val_start = pos + 12; // length of "__LOG__not__"
            // Skip leading spaces
            let mut actual_start = val_start;
            while actual_start < result.len() && result.chars().nth(actual_start) == Some(' ') {
                actual_start += 1;
            }
            let mut val_end = result.len();
            for (idx, ch) in result[actual_start..].char_indices() {
                if ch == ' ' || ch == '(' || ch == ')' || ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^' || ch == ',' {
                    val_end = actual_start + idx;
                    break;
                }
            }
            let val = result[actual_start..val_end].trim();
            let replacement = format!("lnot({})", val);
            result.replace_range(start..val_end, &replacement);
        }

        // Handle binary operators (or, and)
        for op_name in ["or", "and"] {
            let marker = format!("__LOG__{}__", op_name);
            while let Some(pos) = result.find(&marker) {
                let left_end = pos.saturating_sub(1);
                let mut left_start = 0;
                for (idx, ch) in result[..left_end].char_indices().rev() {
                    if ch == ' ' || ch == '(' || ch == ')' || ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^' {
                        left_start = idx + 1;
                        break;
                    }
                    left_start = idx;
                }
                let left = result[left_start..left_end].trim();

                let right_start = pos + marker.len();
                // Skip leading spaces
                let mut actual_start = right_start;
                while actual_start < result.len() && result.chars().nth(actual_start) == Some(' ') {
                    actual_start += 1;
                }
                let mut right_end = result.len();
                for (idx, ch) in result[actual_start..].char_indices() {
                    if ch == ' ' || ch == '(' || ch == ')' || ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^' || ch == ',' {
                        right_end = actual_start + idx;
                        break;
                    }
                }
                let right = result[actual_start..right_end].trim();

                let replacement = format!("l{}({}, {})", op_name, left, right);
                result.replace_range(left_start..right_end, &replacement);
            }
        }

        result
    }

    /// Replace floor division (//) with floor(a / b)
    fn replace_floor_division(expr: &str) -> String {
        let mut result = String::new();
        let chars: Vec<char> = expr.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if i + 1 < chars.len() && chars[i] == '/' && chars[i + 1] == '/' {
                // Find the left operand by scanning backwards in result
                let left_end = result.len();
                let mut paren_depth = 0;
                let mut left_start = left_end;

                for (idx, ch) in result.chars().rev().enumerate() {
                    if ch == ')' {
                        paren_depth += 1;
                    } else if ch == '(' {
                        if paren_depth > 0 {
                            paren_depth -= 1;
                        } else {
                            left_start = left_end - idx - 1;
                            break;
                        }
                    } else if paren_depth == 0 && (ch == '+' || ch == '-' || ch == '|' || ch == '&' || ch == '=' || ch == '>' || ch == '<') {
                        left_start = left_end - idx;
                        break;
                    } else if paren_depth == 0 && idx > 0 && ch == ' ' {
                        // Check if next non-space char is an operator
                        let remaining = &result[..left_end - idx];
                        if remaining.trim().ends_with(|c: char| c == '+' || c == '-' || c == '|' || c == '&' || c == '=' || c == '>' || c == '<') || remaining.trim().is_empty() {
                            left_start = left_end - idx;
                            break;
                        }
                    }
                    left_start = left_end - idx - 1;
                }

                let left_operand = result[left_start..].trim().to_string();
                result.truncate(left_start);

                // Find the right operand by scanning forwards
                let right_start = i + 2;
                let mut right_end = right_start;
                let mut paren_depth = 0;

                while right_end < chars.len() {
                    let ch = chars[right_end];
                    if ch == '(' {
                        paren_depth += 1;
                    } else if ch == ')' {
                        if paren_depth == 0 {
                            break;
                        }
                        paren_depth -= 1;
                    } else if paren_depth == 0 && (ch == '+' || ch == '-' || ch == '*' || ch == '/' || ch == '^' || ch == '|' || ch == '&' || ch == '=' || ch == '>' || ch == '<') {
                        break;
                    }
                    right_end += 1;
                }

                let right_operand: String = chars[right_start..right_end].iter().collect::<String>().trim().to_string();
                result.push_str(&format!("floor({} / {})", left_operand, right_operand));
                i = right_end;
            } else {
                result.push(chars[i]);
                i += 1;
            }
        }

        result
    }

    /// Evaluates a mathematical expression
    pub fn evaluate(&mut self, expression: &str) -> Result<f64> {
        // Preprocess expression for additional operator support
        let expression = Self::preprocess_expression(expression);

        // Check if expression is a mod function call (name(args))
        if let Some(paren_pos) = expression.find('(') {
            let func_name = expression[..paren_pos].trim();
            // We only need to check if the mod exists, we don't need to use the mod_def value here
            if self.mod_manager.get_mod(func_name).is_some() {
                // This is a mod function call
                return self.evaluate_mod(func_name, &expression);
            }
        }

        // Otherwise, evaluate normally
        if self.safe_mode {
            // 使用缓存机制来优化性能
            let cache_key = expression.to_string();
            if let Some(cached_result) = self.get_cached_result(&cache_key) {
                return Ok(cached_result);
            }

            // 使用缓存的表达式对象来优化性能
            let expr = if let Some(cached_expr) = self.expr_cache.get(cache_key.as_str()) {
                cached_expr.clone()
            } else {
                let parsed_expr = expression.parse::<Expr>()?;
                let arc_expr = Arc::new(parsed_expr);
                self.expr_cache.insert(cache_key.clone(), arc_expr.clone());
                arc_expr
            };

            // 评估表达式
            match expr.eval_with_context(&self.context) {
                Ok(result) => {
                    // 缓存结果
                    self.result_cache.insert(cache_key, result);
                    // 更新最近结果 (m 常量)
                    self.set_last_result(result);
                    Ok(result)
                }
                Err(e) => bail!("Evaluation error: {}", e),
            }
        } else {
            // In a real implementation, this would allow more complex expressions
            // For now, we'll just use the same safe evaluation
            match expression.parse::<Expr>() {
                Ok(expr) => match expr.eval_with_context(&self.context) {
                    Ok(result) => {
                        // 更新最近结果 (m 常量)
                        self.set_last_result(result);
                        Ok(result)
                    }
                    Err(e) => bail!("Evaluation error: {}", e),
                },
                Err(e) => bail!("Parse error: {}", e),
            }
        }
    }

    /// Evaluate a mod function call
    fn evaluate_mod(&self, mod_name: &str, expression: &str) -> Result<f64> {
        // Extract function name and arguments
        let paren_start = expression
            .find('(')
            .ok_or_else(|| anyhow::anyhow!("Invalid mod call"))?;
        let paren_end = expression
            .rfind(')')
            .ok_or_else(|| anyhow::anyhow!("Invalid mod call"))?;

        if paren_end <= paren_start {
            bail!("Invalid mod call: empty parentheses");
        }

        let args_str = &expression[paren_start + 1..paren_end];
        let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();

        // Get mod definition
        let mod_def = self
            .mod_manager
            .get_mod(mod_name)
            .ok_or_else(|| anyhow::anyhow!("Mod '{}' not found", mod_name))?;

        // Check number of arguments
        if args.len() != mod_def.var.needvars.len() {
            bail!(
                "Mod '{}' expects {} arguments, got {}",
                mod_name,
                mod_def.var.needvars.len(),
                args.len()
            );
        }

        // Build a new context with the provided arguments
        let mut ctx = self.context.clone();

        // Evaluate each argument and bind to variable
        for (i, var_name) in mod_def.var.needvars.iter().enumerate() {
            let arg_val: f64 = match args[i].parse::<f64>() {
                Ok(v) => v,
                Err(_) => {
                    // Try to evaluate as an expression
                    match args[i].parse::<Expr>() {
                        Ok(expr) => match expr.eval_with_context(&self.context) {
                            Ok(v) => v,
                            Err(e) => bail!("Failed to evaluate argument '{}': {}", args[i], e),
                        },
                        Err(e) => bail!("Failed to parse argument '{}': {}", args[i], e),
                    }
                }
            };

            ctx.var(var_name, arg_val);
        }

        // Get the calculation expression
        let calc_expr = mod_def
            .calc
            .howto
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Mod '{}' has no calculation defined", mod_name))?;

        // Evaluate the calculation expression
        match calc_expr.parse::<Expr>() {
            Ok(expr) => match expr.eval_with_context(&ctx) {
                Ok(result) => Ok(result),
                Err(e) => bail!("Mod calculation error: {}", e),
            },
            Err(e) => bail!("Mod expression parse error: {}", e),
        }
    }

    // 添加缓存机制
    fn get_cached_result(&self, cache_key: &str) -> Option<f64> {
        // 实现缓存逻辑
        self.result_cache.get(cache_key).copied()
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_logical_or() {
        let result = Evaluator::preprocess_expression("1 or 0");
        assert_eq!(result, "lor(1, 0)");
    }

    #[test]
    fn test_preprocess_logical_and() {
        let result = Evaluator::preprocess_expression("1 and 1");
        assert_eq!(result, "land(1, 1)");
    }

    #[test]
    fn test_preprocess_logical_not() {
        let result = Evaluator::preprocess_expression("not 0");
        assert_eq!(result, "lnot(0)");
    }

    #[test]
    fn test_preprocess_comparison_eq() {
        let result = Evaluator::preprocess_expression("5 == 5");
        assert_eq!(result, "_cmp_eq(5, 5)");
    }

    #[test]
    fn test_preprocess_floor_division() {
        let result = Evaluator::preprocess_expression("7 // 2");
        assert_eq!(result, "floor(7 / 2)");
    }
}
