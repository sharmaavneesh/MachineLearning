def fizzbuzz(n):
    # Logic Explanation
    # First condition is testing if the number is divisible by 3 as well as by 5 and returns "FizzBuzz"
    # If the first condition is not satisfied then it checks if it is divisible by 3 and return "Fizz"
    # If both of the above tests are not satisfying, then it will check whether it is divisible by 5 and return "Buzz"
    # If all the conditions above do not satisfy then it returns "Other"
    if n % 3 == 0 and n % 5 == 0:
        return 'FizzBuzz'
    elif n % 3 == 0:
        return 'Fizz'
    elif n % 5 == 0:
        return 'Buzz'
    else:
        return 'Other'