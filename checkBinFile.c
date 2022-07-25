#include <stdio.h>

int main(int argc, char *argv[])
{
    int nlines = 10;

    double buf[5];

    FILE *f = fopen(argv[1], "r");

    int i;
    for(i=0; i<nlines; i++)
    {
        fread(buf, sizeof(double), 5, f);
        printf("%e %e %e %e %e\n", buf[0], buf[1], buf[2], buf[3], buf[4]);
    }

    return 0;
}
