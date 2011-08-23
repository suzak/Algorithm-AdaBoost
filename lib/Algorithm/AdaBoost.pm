package Algorithm::AdaBoost;
use strict;
use warnings;

use Algorithm::AdaBoost::Classifier qw(POSITIVE NEGATIVE NOT_APPLICABLE);
use Carp ();
use Exporter::Lite;

our @EXPORT_OK = @Algorithm::AdaBoost::Classifier::EXPORT_OK;


sub new {
    my ($class, $base, @classifiers) = @_;

    @classifiers = map {
        my $classifier = "$base\::$_";
        eval qq<require $classifier>;
        die $@ if $@;
        $classifier->new($_);
    } @classifiers;

    return bless {
        classifiers => \@classifiers,
        data        => [],
    }, $class;
}

sub add_data {
    my ($self, $data, $label) = @_;
    push @{ $self->{data} }, map { +{
        data   => $_,
        label  => $label,
        weight => {},
        score  => {},
    } } @$data,
}

sub train {
    my ($self) = @_;
    for (my $i = 0; $i < @{ $self->{classifiers} }; ++$i) {
        my $c = $self->{classifiers}->[$i];
        my $error = $c->minimize_error($self->{data});
        $c->{weight} = $error == 1 ? 0 :
                       $error == 0 ? 10 : 0.5 * log((1 - $error) / $error));
        return if $i >= @{ $self->{classifiers} } - 1;

        my $normalization_factor = $error ? 2 * sqrt($error * (1 - $error)) : 1;
        my $next = $self->{classifiers}->[$i + 1]->name;
        for (@{ $self->{data} }) {
            if (defined(my $score = $_->{score}->{$c->name})) {
                $_->{weight}->{$next} = $_->{weight}->{$c->name} * exp(- $c->{weight} * $score) / $normalization_factor;
            } else {
                $_->{weight}->{$next} = $_->{weight}->{$c->name};
            }
        }
    }
}

sub init_train {
    my ($self) = @_;

    Carp::croak 'Training set is empty. You must supply it via add_data before.'
        unless @{ $self->{data} };
    my $initial_weight = 1 / @{ $self->{data} };

    my $i = 0;
    for my $c (@{ $self->{classifiers} }) {
        for (@{ $self->{data} }) {
            $_->{score}->{$c->name} = $c->score($_->{data});
            $_->{weight}->{$c->name} = $i ? undef : $initial_weight;
        }
        $i++;
    }
}

sub classify {
    my ($self, $data) = @_;
    return $self->score($data) > 0 ? POSITIVE : NEGATIVE;
}

sub score {
    my ($self, $data) = @_;
    $self->score_explain($data)->score;
}

sub explain {
    my ($self, $data) = @_;
    $self->score_explain($data)->explain;
}

sub score_explain {
    my ($self, $data) = @_;
    my $ret = {
        score   => 0,
        explain => {},
    };
    for (@{ $self->{classifiers} }) {
        $ret->{score} += (my $score = $_->weight * $_->classify($data));
        $ret->{explain}->{$_->EXPLAIN || $_->name} = $score;
    }
    return Algorithm::AdaBoost::Result->new(%$ret);
}

sub load {
    my ($self, $params) = @_;
    for (@{ $self->{classifiers} }) {
        $_->weight($params->{$_->name}->{weight});
        $_->division_pos($params->{$_->name}->{division_pos});
    }
}

sub dump {
    my ($self) = @_;
    return +{ map {
        $_->name => {
            weight       => $_->weight,
            division_pos => $_->division_pos,
        },
    } @{ $self->{classifiers} } };
}

sub status {
    my ($self) = @_;
    join "\n\n", (map {
        my $c = $_;
        my $h;
        push @{ $h->{int($_->{score}->{$c->name} * 10)} }, $_
            for sort {
                $a->{score}->{$c->name} <=> $b->{score}->{$c->name}
            } grep { defined $_->{score}->{$c->name} } @{ $self->{data} };

        join "\n", 'Histogram of ' . $c->name, (map {
            sprintf "    %1.1f|%s", $_ / 10, join '', map {
                $_->{label} == POSITIVE ? '+' :
                $_->{label} == NEGATIVE ? '-' : ''
            } @{ $h->{$_} || [] }
        } 0..10),
        sprintf("Division Pos: %1.4f", $c->division_pos),
        sprintf("Mininum Error: %1.4f", $c->minimum_error),
        sprintf("Weight: %1.4f", $c->weight);
    } @{ $self->{classifiers} }),
    sprintf "Similarity: %1.4f", (grep { $_->{label} eq $self->classify($_->{data}) } @{ $self->{data} }) / @{ $self->{data} };
}


package
    Algorithm::AdaBoost::Result;

use Class::Accessor::Lite
    new => 1,
    ro => [qw( score explain )];


package
    Algorithm::AdaBoost;


1;
__END__
