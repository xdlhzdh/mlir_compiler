#include <iostream>

#include "interpreter_v4.hpp"

using namespace Interpreter_V4;

void runExample(const std::string &title, const std::string &code) {
  std::cout << "\n======= " << title << " =======\n";
  std::cout << "Code:\n" << code << "\n";
  std::cout << "Output:\n";

  try {
    auto env = std::make_shared<Env>();
    Parser p(code);
    auto stmts = p.parseProgram();
    for (auto &stmt : stmts)
      stmt->exec(env);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
}

int main() {
  std::cout << "===== Compiler V4 Demo: Break and Continue Support =====\n";

  // 1. Basic Break Statement
  runExample("Basic Break - Exit Loop Early", R"(
let i = 0;
while (i < 10) {
    if (i == 5) {
        print "Breaking at", i;
        break;
    }
    print i;
    i += 1;
}
print "Loop ended";
)");

  // 2. Basic Continue Statement
  runExample("Basic Continue - Skip Even Numbers", R"(
let i = 0;
while (i < 10) {
    i += 1;
    if (i % 2 == 0) {
        continue;
    }
    print "Odd:", i;
}
print "Loop completed";
)");

  // 3. Break in Nested Loops
  runExample("Break in Nested Loops", R"(
let i = 0;
while (i < 3) {
    print "Outer loop:", i;
    let j = 0;
    while (j < 5) {
        if (j == 3) {
            print "  Breaking inner loop at j =", j;
            break;
        }
        print "  Inner loop:", j;
        j += 1;
    }
    i += 1;
}
print "All loops ended";
)");

  // 4. Continue in Nested Loops
  runExample("Continue in Nested Loops", R"(
let i = 0;
while (i < 3) {
    i += 1;
    if (i == 2) {
        print "Skipping outer iteration", i;
        continue;
    }
    print "Outer:", i;
    let j = 0;
    while (j < 3) {
        j += 1;
        if (j == 2) {
            continue;
        }
        print "  Inner:", j;
    }
}
)");

  // 5. Break with Condition - Find First Match
  runExample("Break - Find First Even Square", R"(
let i = 1;
let found = 0;
while (i < 20) {
    let square = i * i;
    if (square % 2 == 0) {
        print "Found first even square:", square, "from", i;
        found = square;
        break;
    }
    i += 1;
}
if (found > 0) {
    print "Result:", found;
} else {
    print "Not found";
}
)");

  // 6. Continue with Complex Condition
  runExample("Continue - Filter and Process", R"(
let i = 1;
let sum = 0;
while (i <= 10) {
    if (i % 3 == 0) {
        print "Skipping multiple of 3:", i;
        i += 1;
        continue;
    }
    sum += i;
    print "Adding", i, "sum is now", sum;
    i += 1;
}
print "Final sum:", sum;
)");

  // 7. Break and Continue Together
  runExample("Break and Continue Together", R"(
let count = 0;
let i = 0;
while (i < 20) {
    i += 1;
    
    if (i % 2 == 0) {
        continue;
    }
    
    if (i > 15) {
        print "Stopping at", i;
        break;
    }
    
    print "Processing:", i;
    count += 1;
}
print "Processed", count, "items";
)");

  // 8. Break in Function with Loop
  runExample("Break in Function", R"(
fn findDivisor(n) {
    let i = 2;
    while (i < n) {
        if (n % i == 0) {
            return i;
        }
        i += 1;
    }
    return n;
}

print "First divisor of 15:", findDivisor(15);
print "First divisor of 17:", findDivisor(17);
print "First divisor of 20:", findDivisor(20);
)");

  // 9. Continue with Early Increment
  runExample("Continue - Print Only Primes (Simplified)", R"(
fn isPrime(n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    let i = 3;
    while (i * i <= n) {
        if (n % i == 0) {
            return false;
        }
        i += 2;
    }
    return true;
}

let num = 2;
let count = 0;
while (num < 30) {
    if (!isPrime(num)) {
        num += 1;
        continue;
    }
    print "Prime:", num;
    count += 1;
    num += 1;
}
print "Found", count, "primes";
)");

  // 10. Complex Example - Search with Break and Continue
  runExample("Complex - Search and Filter", R"(
fn processNumbers() {
    let i = 1;
    let results = 0;
    
    while (i <= 50) {
        if (i % 5 == 0) {
            i += 1;
            continue;
        }
        
        if (i % 3 == 0) {
            results += 1;
            print "Found:", i;
        }
        
        if (results >= 5) {
            print "Found enough results";
            break;
        }
        
        i += 1;
    }
    
    return results;
}

let total = processNumbers();
print "Total results:", total;
)");

  // 11. Break with Closure
  runExample("Break with Closure - Search in Closure", R"(
fn makeSearcher(max) {
    fn search(target) {
        let i = 0;
        while (i < max) {
            if (i * i == target) {
                return i;
            }
            if (i * i > target) {
                break;
            }
            i += 1;
        }
        return -1;
    }
    return search;
}

let searcher = makeSearcher(100);
print "Square root of 25:", searcher(25);
print "Square root of 49:", searcher(49);
print "Square root of 30:", searcher(30);
)");

  // 12. Ternary with Break/Continue Context
  runExample("Ternary Operator with Loop Control", R"(
let i = 0;
let evens = 0;
let odds = 0;
while (i < 15) {
    i += 1;
    let isEven = i % 2 == 0;
    let category = isEven ? "even" : "odd";
    
    if (i > 10) {
        print "Stopping at", i;
        break;
    }
    
    if (isEven) {
        evens += 1;
    } else {
        odds += 1;
    }
    
    print i, "is", category;
}
print "Evens:", evens, "Odds:", odds;
)");

  return 0;
}
